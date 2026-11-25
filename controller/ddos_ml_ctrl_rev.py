#!/usr/bin/env python3
"""
ddos_ml_ctrl.py

Working controller for cybersecure_sdn:
- installs L3 forwarding (ipv4_lpm)
- installs spoof_guard wildcard allow + specific host bindings
- enables ARP flooding (best effort)
- enables telemetry_ctl default action (best effort)
- keeps controller alive for future digest support

Run with:
    python3 controller/ddos_ml_ctrl.py
"""

import os
import sys
import csv
import time
import ipaddress
import threading
from pathlib import Path
from p4.v1 import p4runtime_pb2
import torch

ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_DIR = Path(__file__).resolve().parent
P4RUNTIME_LIB_DIR = CONTROLLER_DIR / "p4runtime_lib"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CONTROLLER_DIR))
sys.path.insert(0, str(P4RUNTIME_LIB_DIR))

from p4runtime_lib.helper import P4InfoHelper
import p4runtime_lib.bmv2
from p4runtime_lib.switch import ShutdownAllSwitchConnections
from policy_manager import apply_mitigation        #added
from digest_handler_v2 import handle_digest, start as rev_start, CFG as REV_CFG   # REV
from dnn_model import load_model         #addedformitigation



BUILD = ROOT / "build"
P4INFO_FILE_PATH = os.environ.get("P4INFO", str(BUILD / "ddos_detect_rev.p4info.txt"))   # REV
JSON_FILE_PATH = os.environ.get("P4JSON", str(BUILD / "ddos_detect_rev.json"))   # REV

LOG_DIR = CONTROLLER_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_CSV = LOG_DIR / "flows.csv"

HOSTS = {
    "10.0.1.1": ("s1", 1),
    "10.0.1.2": ("s1", 2),
    "10.0.2.1": ("s2", 1),
    "10.0.2.2": ("s2", 2),
    "10.0.3.1": ("s3", 1),
    "10.0.3.2": ("s3", 2),
    "10.0.4.1": ("s4", 1),
    "10.0.4.2": ("s4", 2),
}

SWITCHES = {
    "s1": {"address": "127.0.0.1:50051", "device_id": 1},
    "s2": {"address": "127.0.0.1:50052", "device_id": 2},
    "s3": {"address": "127.0.0.1:50053", "device_id": 3},
    "s4": {"address": "127.0.0.1:50054", "device_id": 4},
}

RIGHT_PORT = {"s1": 3, "s2": 4, "s3": 4}
LEFT_PORT = {"s2": 3, "s3": 3, "s4": 3}

DATA_PORTS = {
    "s1": [1, 2, 3],
    "s2": [1, 2, 3, 4],
    "s3": [1, 2, 3, 4],
    "s4": [1, 2, 3],
}

ROUTER_MAC = {
    "s1": "00:aa:00:00:01:00",
    "s2": "00:aa:00:00:02:00",
    "s3": "00:aa:00:00:03:00",
    "s4": "00:aa:00:00:04:00",
}

TABLE_IPV4_LPM = "MyIngress.ipv4_lpm"
ACTION_IPV4_FORWARD = "MyIngress.ipv4_forward"

TABLE_ARP_CTL = "MyIngress.arp_ctl"
ACTION_SET_MCAST = "MyIngress.set_mcast_grp"

TABLE_SPOOF = "MyIngress.spoof_guard"
ACTION_ALLOW_SRC = "MyIngress.allow_src"

TABLE_TELEMETRY = "MyIngress.telemetry_ctl"
ACTION_ENABLE_DIGEST = "MyIngress.enable_digest"

TABLE_SRC_RATE_LIMIT = "MyIngress.src_rate_limit" #addednew
ACTION_RL_MARK = "MyIngress.rl_mark"
ACTION_RL_NOP = "MyIngress.rl_nop"

MGID = 1
ENABLE_MITIGATION = False        #added
ATTACKER_IP = "10.0.2.2"


def mac_to_int(mac_str: str) -> int:
    return int(mac_str.replace(":", ""), 16)


def host_mac_from_ip(ip: str) -> str:
    parts = ip.split(".")
    i = int(parts[2])
    j = int(parts[3])
    return f"00:00:00:00:{i:02x}:{j:02x}"


def next_hop_port(current_sw: str, dst_sw: str, dst_host_port: int) -> int:
    if current_sw == dst_sw:
        return dst_host_port
    ci, di = int(current_sw[1:]), int(dst_sw[1:])
    return RIGHT_PORT[current_sw] if di > ci else LEFT_PORT[current_sw]


def write_csv_header_if_needed():
    if not LOG_CSV.exists():
        with open(LOG_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "device_id", "src_ip", "dst_ip", "protocol",
                "src_port", "dst_port", "tag_susp", "timestamp"
            ])


def append_digest_row(device_id, src_ip, dst_ip, proto, sport, dport, tag):
    write_csv_header_if_needed()
    with open(LOG_CSV, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            device_id,
            str(ipaddress.IPv4Address(src_ip)),
            str(ipaddress.IPv4Address(dst_ip)),
            proto, sport, dport, tag, int(time.time())
        ])


def write_ipv4_routes(p4info_helper, sw_conn, sw_name):
    src_mac = ROUTER_MAC[sw_name]

    for ip, (dst_sw, dst_host_port) in HOSTS.items():
        out_port = next_hop_port(sw_name, dst_sw, dst_host_port)
        dst_mac = host_mac_from_ip(ip)

        entry = p4info_helper.buildTableEntry(
            table_name=TABLE_IPV4_LPM,
            match_fields={
                "hdr.ipv4.dstAddr": (ip, 32)
            },
            action_name=ACTION_IPV4_FORWARD,
            action_params={
                "dst_mac": mac_to_int(dst_mac),
                "src_mac": mac_to_int(src_mac),
                "port": out_port
            }
        )
        sw_conn.WriteTableEntry(entry)
        print(f"[{sw_name}] LPM {ip}/32 -> port {out_port} dst_mac {dst_mac}")


def enable_arp_flooding(p4info_helper, sw_conn, sw_name):
    try:
        replicas = [{"egress_port": p, "instance": 0} for p in DATA_PORTS[sw_name]]
        mg_entry = p4info_helper.buildMulticastGroupEntry(MGID, replicas)
        sw_conn.WritePREEntry(mg_entry)

        arp_default = p4info_helper.buildTableEntry(
            table_name=TABLE_ARP_CTL,
            default_action=True,
            action_name=ACTION_SET_MCAST,
            action_params={"mgid": MGID}
        )
        sw_conn.WriteTableEntry(arp_default)
        print(f"[{sw_name}] ARP flood enabled, MGID {MGID} ports {DATA_PORTS[sw_name]}")
    except Exception as e:
        print(f"[{sw_name}] ARP flood setup skipped: {e}")


# def install_spoof_allow_all(p4info_helper, sw_conn, sw_name):
#     try:
#         entry = p4info_helper.buildTableEntry(
#             table_name=TABLE_SPOOF,
#             match_fields={},
#             action_name=ACTION_ALLOW_SRC,
#             action_params={}
#         )
#         sw_conn.WriteTableEntry(entry)
#         print(f"[{sw_name}] spoof_guard wildcard allow installed")
#     except Exception as e:
#         print(f"[{sw_name}] spoof_guard wildcard install failed: {e}")


def write_spoof_bindings(p4info_helper, sw_conn, sw_name):               #commentedformitigation 
    for ip, (h_sw, h_port) in HOSTS.items():
        if h_sw != sw_name:
            continue

        mac = host_mac_from_ip(ip)

        entry = p4info_helper.buildTableEntry(
            table_name=TABLE_SPOOF,
            match_fields={
                "standard_metadata.ingress_port": h_port,
                "hdr.ethernet.srcAddr": mac_to_int(mac),
                "hdr.ipv4.srcAddr": ip,
            },
            action_name=ACTION_ALLOW_SRC,
            action_params={}
        )
        sw_conn.WriteTableEntry(entry)
        print(f"[{sw_name}] spoof binding {ip} @port {h_port} MAC {mac}")


#Addednew
def write_src_rate_limit_rule(p4info_helper, sw_conn, sw_name, attacker_ip):
    try:
        # only install on the switch directly connected to the attacker
        host_info = HOSTS.get(attacker_ip)
        if not host_info:
            return

        attacker_sw, _ = host_info
        if attacker_sw != sw_name:
            return

        entry = p4info_helper.buildTableEntry(
            table_name=TABLE_SRC_RATE_LIMIT,
            match_fields={
                "hdr.ipv4.srcAddr": attacker_ip
            },
            action_name=ACTION_RL_MARK,
            action_params={}
        )
        sw_conn.WriteTableEntry(entry)
        print(f"[{sw_name}] rate-limit rule installed for attacker {attacker_ip}")
    except Exception as e:
        print(f"[{sw_name}] failed to install rate-limit rule for {attacker_ip}: {e}")

# def write_spoof_bindings(p4info_helper, sw_conn, sw_name):
#     for ip, (h_sw, h_port) in HOSTS.items():
#         if h_sw != sw_name:
#             continue

#         mac = host_mac_from_ip(ip)

#         action = ACTION_ALLOW_SRC
#         if ENABLE_MITIGATION and ip == ATTACKER_IP:
#             action = "MyIngress.drop"
#             print(f"[{sw_name}] Mitigation active: blocking attacker {ip} at port {h_port}")

#         entry = p4info_helper.buildTableEntry(
#             table_name=TABLE_SPOOF,
#             match_fields={
#                 "standard_metadata.ingress_port": h_port,
#                 "hdr.ethernet.srcAddr": mac_to_int(mac),
#                 "hdr.ipv4.srcAddr": ip,
#             },
#             action_name=action,
#             action_params={}
#         )
#         sw_conn.WriteTableEntry(entry)

#         if action == ACTION_ALLOW_SRC:
#             print(f"[{sw_name}] spoof binding {ip} @port {h_port} MAC {mac}")
#         else:
#             print(f"[{sw_name}] DROP binding {ip} @port {h_port} MAC {mac}")

def enable_telemetry_wildcard(p4info_helper, sw_conn, sw_name):
    try:
        entry = p4info_helper.buildTableEntry(
            table_name=TABLE_TELEMETRY,
            default_action=True,
            action_name=ACTION_ENABLE_DIGEST,
            action_params={}
        )
        sw_conn.WriteTableEntry(entry)
        print(f"[{sw_name}] telemetry_ctl default action installed (enable digest)")
    except Exception as e:
        print(f"[{sw_name}] telemetry setup skipped: {e}")


def register_digest_callback(sw_conn, sw_name, model, device):         #addedformitigation
    def digest_cb(digest_msg):
        # REV: per-digest print removed (it slowed the stream thread under load)
        handle_digest(digest_msg, sw_conn, model, device)

    try:
        if hasattr(sw_conn, "RegisterDigestCallback"):
            sw_conn.RegisterDigestCallback(digest_cb)
            print(f"[{sw_name}] Registered digest callback")
        elif hasattr(sw_conn, "RegisterDigestHandler"):
            sw_conn.RegisterDigestHandler(digest_cb)
            print(f"[{sw_name}] Registered digest handler")
        else:
            print(f"[{sw_name}] No digest-register method found on switch object; you may need to poll or adapt.")
    except Exception as e:
        print(f"[{sw_name}] Exception while registering digest callback: {e}")

# def digest_poll_loop(sw_conn, sw_name, model, device):
#     """
#     Poll P4Runtime stream channel for digest messages.
#     In this p4runtime_lib version, stream_msg_resp is a gRPC iterator,
#     not a queue, so we iterate over it directly.
#     """
#     print(f"[{sw_name}] Digest polling thread started")

#     while True:
#         try:
#             for response in sw_conn.stream_msg_resp:
#                 if response is None:
#                     continue

#                 if response.HasField("digest"):
#                     print(f"[{sw_name}] Received digest")

#                     # Send digest to ML handler
#                     handle_digest(response.digest, sw_conn, model, device)

#                     # Try ACK if supported
#                     try:
#                         if hasattr(sw_conn, "AckDigestList"):
#                             sw_conn.AckDigestList(
#                                 response.digest.digest_id,
#                                 response.digest.list_id
#                             )
#                     except Exception as ack_error:
#                         print(f"[{sw_name}] Digest ACK skipped: {ack_error}")

#         except Exception as e:
#             print(f"[{sw_name}] Digest polling error: {e}")
#             time.sleep(1)

# def start_digest_polling(sw_conn, sw_name, model, device):
#     t = threading.Thread(
#         target=digest_poll_loop,
#         args=(sw_conn, sw_name, model, device),
#         daemon=True
#     )
#     t.start()
#     print(f"[{sw_name}] Digest polling registered")


def get_digest_id(p4info_helper, digest_name_substring="feature_digest"):
    """
    Find digest ID from P4Info.
    The actual P4Info name may be feature_digest_t or MyIngress.feature_digest_t.
    """
    for d in p4info_helper.p4info.digests:
        name = d.preamble.name
        if digest_name_substring in name:
            print(f"[DigestConfig] Found digest: {name}, id={d.preamble.id}")
            return d.preamble.id

    print("[DigestConfig] Available digests:")
    for d in p4info_helper.p4info.digests:
        print(f"  {d.preamble.name} id={d.preamble.id}")

    raise RuntimeError("Could not find feature_digest in P4Info")


def enable_p4runtime_digest(p4info_helper, sw_conn, sw_name):
    """
    Install P4Runtime DigestEntry so BMv2 sends digest messages.
    Important: WriteRequest must include election_id, otherwise BMv2 says 'Not primary'.
    """
    digest_id = get_digest_id(p4info_helper)

    digest_entry = p4runtime_pb2.DigestEntry()
    digest_entry.digest_id = digest_id
    digest_entry.config.max_timeout_ns = 1000000      # 1 ms
    digest_entry.config.max_list_size = 1             # send each digest quickly
    digest_entry.config.ack_timeout_ns = 0

    req = p4runtime_pb2.WriteRequest()
    req.device_id = sw_conn.device_id

    # VERY IMPORTANT: add election_id, otherwise Write gives "Not primary"
    if hasattr(sw_conn, "election_id"):
        req.election_id.CopyFrom(sw_conn.election_id)
    else:
        req.election_id.high = 0
        req.election_id.low = 1

    update = req.updates.add()
    update.type = p4runtime_pb2.Update.INSERT
    update.entity.digest_entry.CopyFrom(digest_entry)

    try:
        sw_conn.client_stub.Write(req)
        print(f"[{sw_name}] P4Runtime DigestEntry installed, digest_id={digest_id}")
    except Exception as e:
        msg = str(e)
        if "ALREADY_EXISTS" in msg or "already exists" in msg:
            print(f"[{sw_name}] DigestEntry already exists, continuing")
        else:
            raise


def send_digest_ack(sw_conn, digest_msg):
    """
    ACK digest list back to switch.
    """
    req = p4runtime_pb2.StreamMessageRequest()
    req.digest_ack.digest_id = digest_msg.digest_id
    req.digest_ack.list_id = digest_msg.list_id
    sw_conn.stream_msg_req.put(req)


def digest_poll_loop(sw_conn, sw_name, model, device):
    """
    Read P4Runtime StreamChannel for digest messages.
    """
    print(f"[{sw_name}] Digest polling thread started")

    try:
        for response in sw_conn.stream_msg_resp:
            update_type = response.WhichOneof("update")
            print(f"[{sw_name}] Stream message received: {update_type}")


            if update_type == "digest":
                print(f"[{sw_name}] Received digest")
                handle_digest(response.digest, sw_conn, model, device)

                try:
                    send_digest_ack(sw_conn, response.digest)
                except Exception as e:
                    print(f"[{sw_name}] Digest ACK failed: {e}")

            elif update_type == "error":
                print(f"[{sw_name}] Stream error: {response.error}")

    except Exception as e:
        print(f"[{sw_name}] Digest polling stopped/error: {e}")


def start_digest_polling(sw_conn, sw_name, model, device):
    t = threading.Thread(
        target=digest_poll_loop,
        args=(sw_conn, sw_name, model, device),
        daemon=True
    )
    t.start()
    print(f"[{sw_name}] Digest polling registered")

def install_device_id_default(p4info_helper, sw_conn, sw_name, device_id):
    try:
        entry = p4info_helper.buildTableEntry(
            table_name="MyIngress.dev_id_ctl",
            default_action=True,
            action_name="MyIngress.set_device_id",
            action_params={
                "dev_id": device_id
            }
        )
        sw_conn.WriteTableEntry(entry)
        print(f"[{sw_name}] device_id set to {device_id}")
    except Exception as e:
        print(f"[{sw_name}] device_id setup skipped: {e}")

def main():
    p4info_helper = P4InfoHelper(P4INFO_FILE_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device=device)
    connections = {}

    try:
        for sw_name, info in SWITCHES.items():
            print(f"[controller] connecting to {sw_name} @ {info['address']}")

            sw = p4runtime_lib.bmv2.Bmv2SwitchConnection(
                name=sw_name,
                address=info["address"],
                device_id=info["device_id"],
                proto_dump_file=str(LOG_DIR / f"{sw_name}-p4runtime-requests.txt"),
            )
            sw.p4info_helper = p4info_helper #added
            
            sw.MasterArbitrationUpdate()
            sw.SetForwardingPipelineConfig(
                p4info=p4info_helper.p4info,
                bmv2_json_file_path=JSON_FILE_PATH
            )

            print(f"[controller] P4 program installed on {sw_name}")
            connections[sw_name] = sw

            install_device_id_default(p4info_helper, sw, sw_name, info["device_id"])
            enable_p4runtime_digest(p4info_helper, sw, sw_name)

            enable_arp_flooding(p4info_helper, sw, sw_name)

            # install_spoof_allow_all(p4info_helper, sw, sw_name)
            write_ipv4_routes(p4info_helper, sw, sw_name)
            write_spoof_bindings(p4info_helper, sw, sw_name)

            #addednew
            # REV: identity-based source marking (src_rate_limit on 10.0.2.2) removed — it embedded
            # the attacker identity in the data plane (Reviewer 3 #15).
            if REV_CFG["mode"] in ("tag_only", "hybrid_tagdrop"):   # drop tagged packets in the switch
                sw.WriteTableEntry(p4info_helper.buildTableEntry(
                    table_name="MyIngress.tag_policy", default_action=True,
                    action_name="MyIngress.drop", action_params={}))
                print(f"[{sw_name}] REV tag_policy default = drop (tag-only baseline)")


            if REV_CFG.get("telemetry", True):   # REV: S14 measures forwarding with digests off
                enable_telemetry_wildcard(p4info_helper, sw, sw_name)
            else:
                print(f"[{sw_name}] REV telemetry/digests DISABLED for this run")
            # register_digest_callback(sw, sw_name)                         #commentedformitigation 
            register_digest_callback(sw, sw_name, model, device)            #addedformitigation
            # if sw_name == "s2":
            #     start_digest_polling(sw, sw_name, model, device)
            # start_digest_polling(sw, sw_name, model, device)

            # if sw_name == "s2":            #addedformitigation
            #     print("[TEST] Installing manual mitigation for h22 -> h41")
            #     apply_mitigation(
            #         sw,
            #         "10.0.2.2",   # attacker h22
            #         "10.0.4.1",   # victim h41
            #         6,            # TCP
            #         0,            # sport seen in your pipeline for non-parsed traffic
            #         80            # destination port used by hping3
            #     )

        rev_start(device)   # REV: worker thread, live telemetry, rule housekeeping
        print("[controller] Bootstrapping complete. Listening for digests; press Ctrl+C to exit.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("Interrupted by user, shutting down")
    except Exception as e:
        print("Controller error:", e)
        raise
    finally:
        ShutdownAllSwitchConnections()


if __name__ == "__main__":
    main()