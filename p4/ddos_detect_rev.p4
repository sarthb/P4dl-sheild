// P4-DL Shield — REVISION version of ddos_detect.p4 (changes marked REV)
#include <core.p4>
#include <v1model.p4>

/* =========================
 * Headers
 * ========================= */
header ethernet_t {
    bit<48> dstAddr;
    bit<48> srcAddr;
    bit<16> etherType;
}

header ipv4_t {
    bit<4>  version;
    bit<4>  ihl;
    bit<8>  diffserv;      // DSCP(6) + ECN(2)
    bit<16> totalLen;
    bit<16> identification;
    bit<3>  flags;
    bit<13> fragOffset;
    bit<8>  ttl;
    bit<8>  protocol;
    bit<16> hdrChecksum;
    bit<32> srcAddr;
    bit<32> dstAddr;
}

header tcp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<32> seqNo;
    bit<32> ackNo;
    bit<4>  dataOffset;
    bit<3>  res;
    bit<9>  flags;   // NS|CWR|ECE|URG|ACK|PSH|RST|SYN|FIN
    bit<16> window;
    bit<16> checksum;
    bit<16> urgentPtr;
}

header udp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<16> len;
    bit<16> checksum;
}

// Minimal ARP (IPv4)
header arp_t {
    bit<16> htype;
    bit<16> ptype;
    bit<8>  hlen;
    bit<8>  plen;
    bit<16> oper;
    bit<48> sha;   // sender hw addr
    bit<32> spa;   // sender proto addr
    bit<48> tha;   // target hw addr
    bit<32> tpa;   // target proto addr
}

/* =========================
 * Metadata & Digest
 * ========================= */
struct feature_md_t {
    bit<8>  send_digest;   // 1 = export features
    bit<32> device_id;     // set by control-plane
    bit<16> sport;
    bit<16> dport;
    // Optional placeholder if you later emulate meter color in metadata
    bit<2>  meter_color;   // 0/1/2 -> green/yellow/red (unused for now)
    bit<1>  rate_limit_hit;
    bit<1> do_drop; 
    bit<8>  tag;          // REV: behavioural burst tag (no host identity)
    bit<32> burst_cnt;    // REV: packets of this (src,dst) pair in current window
}

struct metadata_t {
    feature_md_t feat;
}

struct headers {
    ethernet_t ethernet;
    ipv4_t     ipv4;
    tcp_t      tcp;
    udp_t      udp;
    arp_t      arp;
}

/* Digest payload to controller (for ML CSV pipeline) */
struct feature_digest_t {
    bit<32> device_id;
    bit<32> src_ip;
    bit<32> dst_ip;
    bit<8>  protocol;
    bit<16> src_port;
    bit<16> dst_port;
    bit<8>  tag_susp;   // 1 if tagged suspicious 
    bit<16> pkt_len;
    bit<48> ingress_ts;   // REV: BMv2 ingress_global_timestamp (us) for latency breakdown
    bit<32> burst_cnt;    // REV: burst counter value that produced the tag
}

/* =========================
 * Parser
 * ========================= */
parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata_t meta,
                inout standard_metadata_t standard_metadata) {
    state start {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            0x0800: parse_ipv4;
            0x0806: parse_arp;
            default: accept;
        }
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            6: parse_tcp;
            17: parse_udp;
            default: accept;
        }
    }

    state parse_tcp {
        packet.extract(hdr.tcp);
        transition accept;
    }

    state parse_udp {
        packet.extract(hdr.udp);
        transition accept;
    }

    state parse_arp {
        packet.extract(hdr.arp);
        transition accept;
    }
}

/* =========================
 * Verify checksum (no-op)
 * ========================= */
control MyVerifyChecksum(inout headers hdr, inout metadata_t meta) {
    apply { }
}

/* =========================
 * Ingress
 * ========================= */
control MyIngress(inout headers hdr,
                  inout metadata_t meta,
                  inout standard_metadata_t standard_metadata) {

    // ---------- Constants ----------
    const bit<8>  IP_PROTO_TCP = 6;   // (currently unused)
    const bit<8>  IP_PROTO_UDP = 17;  // (currently unused)
    // DSCP=0x2A (101010b) -> diffserv = 0xA8 with ECN=0
    const bit<8>  DIFFSERV_SUSP = 0xA8;

    // REV: burst-tag rule. A (src,dst) pair is tagged suspicious when it sends more than
    // BURST_PKTS packets inside one BURST_WIN_US window. Report these two values in the paper.
    const bit<48> BURST_WIN_US = 100000;   // 100 ms window
    const bit<32> BURST_PKTS   = 50;       // > 50 pkts / 100 ms  (= 500 pps) -> tag = 1
    register<bit<32>>(4096) burst_cnt_reg;
    register<bit<48>>(4096) burst_win_reg;

    // ---------- Actions ----------
    action set_device_id(bit<32> dev_id) {
        meta.feat.device_id = dev_id;
    }

    action set_ports_from_tcp_udp() {
        if (hdr.tcp.isValid()) {
            meta.feat.sport = hdr.tcp.srcPort;
            meta.feat.dport = hdr.tcp.dstPort;
        } else if (hdr.udp.isValid()) {
            meta.feat.sport = hdr.udp.srcPort;
            meta.feat.dport = hdr.udp.dstPort;
        } else {
            meta.feat.sport = 0;
            meta.feat.dport = 0;
        }
    }

    // L3 forward: rewrite L2 + choose egress port
    action ipv4_forward(bit<48> dst_mac, bit<48> src_mac, bit<9> port) {
        hdr.ethernet.dstAddr = dst_mac;
        hdr.ethernet.srcAddr = src_mac;
        standard_metadata.egress_spec = port;
    }

    // Multicast flood for ARP (BMv2 PRE group)
    action set_mcast_grp(bit<16> mgid) {
        standard_metadata.mcast_grp = mgid;
    }

    action drop() {
    meta.feat.do_drop = 1;
    mark_to_drop(standard_metadata);
    }

    action tag_suspicious() {
        hdr.ipv4.diffserv = DIFFSERV_SUSP;
    }

    action allow() {
        // no-op; rely on ipv4_lpm or existing egress_spec
    }

    action export_features(bit<8> tag_val) {
        feature_digest_t d;
        d.device_id = meta.feat.device_id;
        d.src_ip    = hdr.ipv4.srcAddr;
        d.dst_ip    = hdr.ipv4.dstAddr;
        d.protocol  = hdr.ipv4.protocol;
        d.src_port  = meta.feat.sport;
        d.dst_port  = meta.feat.dport;
        d.tag_susp  = tag_val;

        d.pkt_len   = hdr.ipv4.totalLen;
        d.ingress_ts = standard_metadata.ingress_global_timestamp;
        d.burst_cnt = meta.feat.burst_cnt;
        
        digest<feature_digest_t>(1, d);  // 1 = arbitrary digest id
    }

    // ----Added---
    action rl_mark() {
    meta.feat.rate_limit_hit = 1;
    }

    action rl_nop() {
    meta.feat.rate_limit_hit = 0;
    }

    // ---------- Tables ----------

    // (A) Spoof guard: bind ingress_port + srcMAC + srcIP
    action allow_src() { }
    table spoof_guard {
        key = {
            standard_metadata.ingress_port : exact;
            hdr.ethernet.srcAddr           : exact;
            hdr.ipv4.srcAddr               : exact;
        }
        actions = { allow_src; drop; NoAction; }
        size = 16384;
        default_action = allow_src();
    }

    // (B) Control-plane guard (no meter field in v1model standard_metadata)
    action ctrl_guard_apply() {
        // For now, simply allow; you can tag/drop via controller rules if needed
        allow();
    }
    table ctrl_plane_guard {
        key = {
            hdr.ipv4.dstAddr : exact;   // controller IPs
        }
        actions = { ctrl_guard_apply; NoAction; }
        size = 8;
        default_action = NoAction();
    }

    // (C) Data-plane flow guard (5-tuple) — placeholder actions; logic via controller
    action flow_guard_apply() {
        // For now, allow; controller can insert tag/drop rules by changing action
        allow();
    }
    table flow_guard {
        key = {
            hdr.ipv4.srcAddr  : exact;
            hdr.ipv4.dstAddr  : exact;
            hdr.ipv4.protocol : exact;
            meta.feat.dport   : exact;
        }
        actions = { flow_guard_apply; drop; NoAction; }
        size = 32768;
        default_action = NoAction();
    }

    // (C2) REV: source-level guard. The controller's flow-churn guard installs a drop here when one
    //      source creates too many new flows per second (control-plane / digest flooding, R4).
    table src_guard {
        key = { hdr.ipv4.srcAddr : exact; }
        actions = { drop; NoAction; }
        size = 1024;
        default_action = NoAction();
    }

    // (D) IPv4 LPM forwarding
    table ipv4_lpm {
        key = { hdr.ipv4.dstAddr: lpm; }
        actions = { ipv4_forward; drop; NoAction; }
        size = 4096;
        default_action = NoAction();
    }

    // (E) ARP flood control
    table arp_ctl {
        actions = { set_mcast_grp; NoAction; }
        size = 1;
        default_action = NoAction();
    }

    // (F) Telemetry control: toggle digest export
    action enable_digest() { meta.feat.send_digest = 1; }
    table telemetry_ctl {
        key = {
            hdr.ipv4.srcAddr  : ternary;
            hdr.ipv4.dstAddr  : ternary;
            hdr.ipv4.protocol : ternary;
        }
        actions = { enable_digest; NoAction; }
        size = 4096;
        default_action = NoAction();
    }

    // (G) Device ID (optional, for digest enrichment)
    table dev_id_ctl {
        actions = { set_device_id; NoAction; }
        size = 1;
        default_action = NoAction();
    }

    // REV: what to do with tagged packets BEFORE the DNN confirms (Reviewer 1 #2 / #5).
    // default NoAction = hybrid mode (tag only informs the DNN);
    // controller sets default to drop() for the "tag-only, in-switch" baseline.
    table tag_policy {
        key = { meta.feat.tag : exact; }
        actions = { drop; NoAction; }
        size = 4;
        default_action = NoAction();
    }

    // ----Added--- 
    table src_rate_limit {
    key = {
        hdr.ipv4.srcAddr : exact;
    }
    actions = {
        rl_mark;
        rl_nop;
        NoAction;
    }
    size = 1024;
    default_action = NoAction();
    }
    
    // ----Added--- 
    register<bit<32>>(1024) rl_count_reg;
    register<bit<32>>(1024) rl_last_ts_reg; 

    action apply_rate_limit() {
    bit<32> idx;
    bit<32> count;
    bit<32> last_ts;
    bit<32> now_ts;

    hash(idx, HashAlgorithm.crc32, 32w0, { hdr.ipv4.srcAddr }, 32w1024);

    rl_count_reg.read(count, idx);
    rl_last_ts_reg.read(last_ts, idx);

    now_ts = (bit<32>) standard_metadata.ingress_global_timestamp;

    // reset bucket roughly every 1 ms
    if ((now_ts - last_ts) > 1000) {
        count = 0;
        last_ts = now_ts;
    }

    count = count + 1;

    rl_count_reg.write(idx, count);
    rl_last_ts_reg.write(idx, last_ts);

    // allow small burst, then drop excess
    if (count > 80) {
        meta.feat.do_drop = 1; 
        mark_to_drop(standard_metadata);
    }
    }


    apply {
        meta.feat.do_drop = 0;
        meta.feat.rate_limit_hit = 0;
        meta.feat.send_digest = 0;
        meta.feat.tag = 0;
        meta.feat.burst_cnt = 0;

        // Optional: set per-switch device id (controller may install default)
        dev_id_ctl.apply();

        // ====================
        // ARP pass-through logic
        // ====================
        // Allow ARP packets (broadcast) to bypass spoof guard
        // so hosts can resolve each other's MACs safely
        if (hdr.ethernet.etherType == 0x0806) {
            hdr.arp.setValid();
            // Apply multicast flood group configured by controller
            arp_ctl.apply();
            // No spoofing / rate limit applied to ARP
            return;
        }

        if (hdr.ipv4.isValid()) {
            // Extract ports into metadata for 5-tuple keys
            set_ports_from_tcp_udp();

            // TTL check + decrement
            if (hdr.ipv4.ttl == 0) { drop(); return; }
            hdr.ipv4.ttl = hdr.ipv4.ttl - 1;

            // 1) Spoof guard
            if (standard_metadata.ingress_port == 1 || standard_metadata.ingress_port == 2) {
                spoof_guard.apply();

                if (meta.feat.do_drop == 1) {
                    return;
                }
            }

            // 2) Control-plane guard
            ctrl_plane_guard.apply();

            if (meta.feat.do_drop == 1) {
                return;
            }

            // 3) Telemetry enable (wildcard/per-flow via controller)
            telemetry_ctl.apply();

            // 3b) REV: behavioural burst tagging (identity-free)
            {
                bit<32> bidx;
                bit<32> bcnt;
                bit<48> bwin;
                bit<48> now = standard_metadata.ingress_global_timestamp;
                hash(bidx, HashAlgorithm.crc32, 32w0, { hdr.ipv4.srcAddr, hdr.ipv4.dstAddr }, 32w4096);
                burst_cnt_reg.read(bcnt, bidx);
                burst_win_reg.read(bwin, bidx);
                if (now - bwin > BURST_WIN_US) {
                    bcnt = 0;
                    bwin = now;
                }
                bcnt = bcnt + 1;
                burst_cnt_reg.write(bidx, bcnt);
                burst_win_reg.write(bidx, bwin);
                meta.feat.burst_cnt = bcnt;
                if (bcnt > BURST_PKTS) { meta.feat.tag = 1; } else { meta.feat.tag = 0; }
            }

            // 3c) REV: source-level guard (flow-churn defence)
            src_guard.apply();
            if (meta.feat.do_drop == 1) {
                return;
            }

            // 4) Per-flow guard.
            // If controller has installed a drop rule, drop() sets meta.feat.do_drop = 1.
            flow_guard.apply();

            if (meta.feat.do_drop == 1) {
                return;
            }

            // 4b) REV: optional pre-confirmation action on tagged packets
            if (meta.feat.tag == 1) {
                tag_policy.apply();
                if (meta.feat.do_drop == 1) {
                    return;
                }
            }

            // 5) Source rate limiter for marked attacker sources
            src_rate_limit.apply();

            if (meta.feat.do_drop == 1) {
                return;
            }

            if (meta.feat.rate_limit_hit == 1 && hdr.tcp.isValid() && hdr.tcp.dstPort == 80) {
                apply_rate_limit();

                if (meta.feat.do_drop == 1) {
                    return;
                }

                if (standard_metadata.egress_spec == 0) {
                    return;
                }
            }

            // 6) Route only if packet was not dropped
            ipv4_lpm.apply();

            // 7) Optional feature export
            if (meta.feat.send_digest == 1) {
                export_features(meta.feat.tag);   // REV: was DIFFSERV==0xA8, which was never set
            }
        }
        // Other EtherTypes implicitly dropped
    }
}

/* =========================
 * Egress (no-op for now)
 * ========================= */
control MyEgress(inout headers hdr,
                 inout metadata_t meta,
                 inout standard_metadata_t standard_metadata) {
    apply { }
}

/* =========================
 * Recompute checksum
 * ========================= */
control MyComputeChecksum(inout headers hdr, inout metadata_t meta) {
    apply {
        update_checksum(
            hdr.ipv4.isValid(),
            { hdr.ipv4.version,
              hdr.ipv4.ihl,
              hdr.ipv4.diffserv,
              hdr.ipv4.totalLen,
              hdr.ipv4.identification,
              hdr.ipv4.flags,
              hdr.ipv4.fragOffset,
              hdr.ipv4.ttl,
              hdr.ipv4.protocol,
              hdr.ipv4.srcAddr,
              hdr.ipv4.dstAddr },
            hdr.ipv4.hdrChecksum,
            HashAlgorithm.csum16
        );
    }
}

/* =========================
 * Deparser
 * ========================= */
control MyDeparser(packet_out packet, in headers hdr) {
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.tcp);
        packet.emit(hdr.udp);
        packet.emit(hdr.arp);
    }
}

/* =========================
 * Pipeline
 * ========================= */
V1Switch(MyParser(),
         MyVerifyChecksum(),
         MyIngress(),
         MyEgress(),
         MyComputeChecksum(),
         MyDeparser()) main;
