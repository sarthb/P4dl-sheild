#!/usr/bin/env python3
"""
make_flow_p4.py -- generate p4/ddos_detect_flow.p4 (PER-FLOW windowed digests) from p4/ddos_detect_rev.p4.

Everything else (parser, spoof guard, burst tag, src_guard, flow_guard, routing) is unchanged.
Only the digest export changes:
  * per-flow registers (slot = crc32(src, dst, proto, dport) mod FLOW_SLOTS) keep the current window
    (start, packets, bytes, last timestamp, max tag) and the flow totals (first timestamp, packets, bytes);
  * a packet whose window has run for >= FLOW_WIN_US closes it: ONE digest (msg_type = 1) with the summary of
    the finished window, then a new window starts with this packet;
  * the first packet of a new flow (or of a flow that takes over a slot) sends ONE digest (msg_type = 0),
    so the controller can still count new flows per source (churn guard).
Windows restart at the first packet after the previous window ended (same logic as the burst tag).

  python3 p4/make_flow_p4.py --window-us 50000          # writes p4/ddos_detect_flow.p4
"""
import argparse, os, re
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.path.join(os.path.dirname(__file__), "ddos_detect_rev.p4"))
ap.add_argument("--dst", default=os.path.join(os.path.dirname(__file__), "ddos_detect_flow.p4"))
ap.add_argument("--window-us", type=int, default=50000)
ap.add_argument("--slots", type=int, default=16384)
A = ap.parse_args()
s = open(A.src).read()

def sub_once(old, new):
    global s
    assert s.count(old) == 1, f"expected exactly one match for: {old[:60]!r} (found {s.count(old)})"
    s = s.replace(old, new)

# 1) metadata fields for the summary
sub_once("""    bit<32> burst_cnt;    // REV: packets of this (src,dst) pair in current window
}""", """    bit<32> burst_cnt;    // REV: packets of this (src,dst) pair in current window
    // REV-FLOW: summary carried to the digest action
    bit<8>  msg_type;     // 0 = new flow, 1 = finished window
    bit<32> f_win_pkts;
    bit<32> f_win_bytes;
    bit<48> f_win_first;
    bit<48> f_win_last;
    bit<8>  f_tag_max;
    bit<48> f_flow_first;
    bit<32> f_cum_pkts;
    bit<32> f_cum_bytes;
}""")

# 2) digest struct (same name, so the controller finds it in P4Info)
m = re.search(r"struct feature_digest_t \{.*?\n\}", s, re.S)
assert m, "feature_digest_t not found"
s = s[:m.start()] + """struct feature_digest_t {      // REV-FLOW: one message per finished flow window (or new flow)
    bit<32> device_id;
    bit<32> src_ip;
    bit<32> dst_ip;
    bit<8>  protocol;
    bit<16> src_port;
    bit<16> dst_port;
    bit<8>  tag_max;          // max burst tag seen in the window
    bit<8>  msg_type;         // 0 = new flow, 1 = finished window
    bit<48> ingress_ts;       // switch time of the packet that triggered the message (us)
    bit<32> win_pkts;
    bit<32> win_bytes;
    bit<48> win_first_ts;
    bit<48> win_last_ts;
    bit<48> flow_first_ts;
    bit<32> cum_pkts;         // flow totals up to and including the window
    bit<32> cum_bytes;
}""" + s[m.end():]

# 3) replace the per-packet export action by the summary action
m = re.search(r"    action export_features\(bit<8> tag_val\) \{.*?digest<feature_digest_t>\(1, d\);[^\n]*\n    \}", s, re.S)
assert m, "export_features action not found"
s = s[:m.start()] + """    action emit_flow_digest() {               // REV-FLOW
        feature_digest_t d;
        d.device_id     = meta.feat.device_id;
        d.src_ip        = hdr.ipv4.srcAddr;
        d.dst_ip        = hdr.ipv4.dstAddr;
        d.protocol      = hdr.ipv4.protocol;
        d.src_port      = meta.feat.sport;
        d.dst_port      = meta.feat.dport;
        d.tag_max       = meta.feat.f_tag_max;
        d.msg_type      = meta.feat.msg_type;
        d.ingress_ts    = standard_metadata.ingress_global_timestamp;
        d.win_pkts      = meta.feat.f_win_pkts;
        d.win_bytes     = meta.feat.f_win_bytes;
        d.win_first_ts  = meta.feat.f_win_first;
        d.win_last_ts   = meta.feat.f_win_last;
        d.flow_first_ts = meta.feat.f_flow_first;
        d.cum_pkts      = meta.feat.f_cum_pkts;
        d.cum_bytes     = meta.feat.f_cum_bytes;
        digest<feature_digest_t>(1, d);
    }""" + s[m.end():]

# 4) per-flow registers next to the burst-tag registers
sub_once("    register<bit<48>>(4096) burst_win_reg;\n", f"""    register<bit<48>>(4096) burst_win_reg;
    // REV-FLOW: per-flow window state, slot = crc32(src, dst, proto, dport) mod FLOW_SLOTS
    const bit<48> FLOW_WIN_US = {A.window_us};
    register<bit<32>>({A.slots}) fl_src;
    register<bit<32>>({A.slots}) fl_dst;
    register<bit<24>>({A.slots}) fl_pd;        // protocol ++ dport
    register<bit<48>>({A.slots}) fl_wstart;
    register<bit<32>>({A.slots}) fl_wpkts;
    register<bit<32>>({A.slots}) fl_wbytes;
    register<bit<48>>({A.slots}) fl_wlast;
    register<bit<8>>({A.slots})  fl_wtag;
    register<bit<48>>({A.slots}) fl_first;
    register<bit<32>>({A.slots}) fl_cpkts;
    register<bit<32>>({A.slots}) fl_cbytes;
""")

# 5) export step: windowed per-flow logic
sub_once("""            if (meta.feat.send_digest == 1) {
                export_features(meta.feat.tag);   // REV: was DIFFSERV==0xA8, which was never set
            }""", f"""            if (meta.feat.send_digest == 1) {{       // REV-FLOW: per-flow windowed export
                bit<32> fi;
                bit<48> now;
                bit<32> plen;
                bit<24> pd;
                bit<32> ksrc;
                bit<32> kdst;
                bit<24> kpd;
                bit<48> wst;
                bit<32> wpk;
                bit<32> wby;
                bit<48> wlast;
                bit<8>  wtag;
                bit<48> ffirst;
                bit<32> cpk;
                bit<32> cby;
                now  = standard_metadata.ingress_global_timestamp;
                plen = (bit<32>) hdr.ipv4.totalLen;
                pd   = hdr.ipv4.protocol ++ meta.feat.dport;
                hash(fi, HashAlgorithm.crc32, 32w0,
                     {{ hdr.ipv4.srcAddr, hdr.ipv4.dstAddr, hdr.ipv4.protocol, meta.feat.dport }}, 32w{A.slots});
                fl_src.read(ksrc, fi);
                fl_dst.read(kdst, fi);
                fl_pd.read(kpd, fi);
                fl_wstart.read(wst, fi);
                fl_wpkts.read(wpk, fi);
                fl_wbytes.read(wby, fi);
                fl_wlast.read(wlast, fi);
                fl_wtag.read(wtag, fi);
                fl_first.read(ffirst, fi);
                fl_cpkts.read(cpk, fi);
                fl_cbytes.read(cby, fi);
                if (wst == 0 || ksrc != hdr.ipv4.srcAddr || kdst != hdr.ipv4.dstAddr || kpd != pd) {{
                    // new flow (or a different flow takes over this slot): one msg_type 0 digest
                    meta.feat.msg_type     = 0;
                    meta.feat.f_win_pkts   = 1;
                    meta.feat.f_win_bytes  = plen;
                    meta.feat.f_win_first  = now;
                    meta.feat.f_win_last   = now;
                    meta.feat.f_tag_max    = meta.feat.tag;
                    meta.feat.f_flow_first = now;
                    meta.feat.f_cum_pkts   = 1;
                    meta.feat.f_cum_bytes  = plen;
                    emit_flow_digest();
                    ksrc = hdr.ipv4.srcAddr;
                    kdst = hdr.ipv4.dstAddr;
                    kpd = pd;
                    wst = now;
                    wpk = 1;
                    wby = plen;
                    wlast = now;
                    wtag = meta.feat.tag;
                    ffirst = now;
                    cpk = 1;
                    cby = plen;
                }} else if (now - wst >= FLOW_WIN_US) {{
                    // window finished: one msg_type 1 digest with its summary, then start a new window
                    meta.feat.msg_type     = 1;
                    meta.feat.f_win_pkts   = wpk;
                    meta.feat.f_win_bytes  = wby;
                    meta.feat.f_win_first  = wst;
                    meta.feat.f_win_last   = wlast;
                    meta.feat.f_tag_max    = wtag;
                    meta.feat.f_flow_first = ffirst;
                    meta.feat.f_cum_pkts   = cpk;
                    meta.feat.f_cum_bytes  = cby;
                    emit_flow_digest();
                    wst = now;
                    wpk = 1;
                    wby = plen;
                    wlast = now;
                    wtag = meta.feat.tag;
                    cpk = cpk + 1;
                    cby = cby + plen;
                }} else {{
                    wpk = wpk + 1;
                    wby = wby + plen;
                    wlast = now;
                    if (meta.feat.tag > wtag) {{ wtag = meta.feat.tag; }}
                    cpk = cpk + 1;
                    cby = cby + plen;
                }}
                fl_src.write(fi, ksrc);
                fl_dst.write(fi, kdst);
                fl_pd.write(fi, kpd);
                fl_wstart.write(fi, wst);
                fl_wpkts.write(fi, wpk);
                fl_wbytes.write(fi, wby);
                fl_wlast.write(fi, wlast);
                fl_wtag.write(fi, wtag);
                fl_first.write(fi, ffirst);
                fl_cpkts.write(fi, cpk);
                fl_cbytes.write(fi, cby);
            }}""")

open(A.dst, "w").write(s)
print(f"wrote {A.dst}: window {A.window_us} us, {A.slots} flow slots")
