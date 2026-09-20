"""List RoCE devices, Linux interfaces, IPs, MTUs and GID indices.

Run without arguments for this host, or with --config FILE to inspect every
node in a deployment over SSH. This is read-only: it does not change .env,
configure networking, open RDMA connections or require downloaded weights.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cluster_doctor
from site_env import config_argument, resolve_config


def suggestions(inventory):
    devices = [row for row in inventory if row["usable"]]
    if not 1 <= len(devices) <= 2:
        return None
    result = {"DGPP_ROCE_DEVICES": " ".join(row["device"] for row in devices)}
    if all(len(row["gids"]) == 1 for row in devices):
        result["DGPP_ROCE_GID_INDICES"] = " ".join(str(row["gids"][0]["index"]) for row in devices)
    return result


def selection(inventory, env):
    """Show the transport's device order, not a guess about cable connectivity."""
    explicit = env.get("DGPP_ROCE_DEVICES", "").split()
    devices = explicit or sorted(row["device"] for row in inventory
                                 if row["port"] == "1" and row["active"])
    indices = env.get("DGPP_ROCE_GID_INDICES", "").split()
    return {"mode": "configured" if explicit else "automatic", "devices": devices,
            "gid_indices": indices or None}


def inspect_cluster(cfg=None):
    """Collect inventories independently of CLI formatting or config writes."""
    if cfg:
        cluster_doctor.require_head(cfg["nodes"][0])
    nodes = cfg["nodes"] if cfg else ["localhost"]
    inventories = {}
    errors = {}
    selections = {}
    for rank, host in enumerate(nodes):
        try:
            if rank == 0:
                inventories[host] = cluster_doctor.roce_inventory()
            else:
                target = f"{cfg['ssh_user']}@{host}" if cfg["ssh_user"] else host
                result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                         target, "python3 - roce --spec '{}'"],
                                        input=Path(cluster_doctor.__file__).read_text(), text=True,
                                        capture_output=True, timeout=30)
                if result.returncode:
                    raise RuntimeError(result.stderr.strip() or "SSH discovery failed")
                inventories[host] = json.loads(result.stdout)
            selections[host] = selection(inventories[host], cfg["node_env"][rank] if cfg else {})
            if cfg and len(cfg["nodes"]) == 1:
                selections[host] = {"mode": "not needed (single node)", "devices": [], "gid_indices": None}
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            errors[host] = str(error)
    return {"inventories": inventories, "selections": selections, "errors": errors}


def print_report(report, nodes):
    inventories, selections, errors = (report[key] for key in ("inventories", "selections", "errors"))
    overrides = {}
    for rank, host in enumerate(nodes):
        if host in errors:
            print(f"\nFAIL {host}: inventory unknown: {errors[host]}")
            continue
        rows = inventories[host]
        print(f"\n{host}: verbs device / port -> Linux interface, IPs, MTU, RoCE-v2 GID")
        chosen = selections[host]
        print(f"  Deployment lane order ({chosen['mode']}): " + (" -> ".join(chosen["devices"]) or "none"))
        if chosen["devices"]:
            print("  GID indices: " + (" ".join(chosen["gid_indices"]) if chosen["gid_indices"] else "automatic; candidates below"))
        if not rows:
            print("  No RDMA devices found. Check the driver and network setup.")
        for row in rows:
            status = "locally eligible" if row["usable"] else "not locally eligible"
            print(f"  {row['device']} / {row['port']}: {status}")
            for gid in row["gids"]:
                print(f"    {gid['interface'] or '?'}  {', '.join(gid['ips']) or gid['address']}  MTU {gid['mtu'] or '?'}  GID index {gid['index']} ({gid['address']})")
        setting = suggestions(rows)
        if setting is None:
            print("  Select one or two active port-1 devices with routable RoCE-v2 GIDs; no settings suggested.")
        elif rank == 0:
            print("  Candidate settings (not a connectivity recommendation):")
            for key, value in setting.items():
                print(f'  {key}="{value}"')
        else:
            overrides[host] = setting
    if overrides:
        print("\nPer-node candidate settings (merge with existing overrides):")
        print("DGPP_NODE_OVERRIDES='" + json.dumps(overrides, separators=(",", ":")) + "'")
    print("\nBefore copying settings: match lane order by subnet across nodes, not by device name.")
    print("Multiple GIDs require choosing the intended network. Discovery does not test RDMA connectivity.")
    if errors:
        print("Some nodes could not be inspected. Fix SSH access from rank 0; omit --config to inspect only this host.")
    return 1 if errors else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=config_argument, help="inspect the first world_size nodes; run on rank 0")
    parser.add_argument("--json", action="store_true", help="print inventories, selections and per-node errors as JSON")
    args = parser.parse_args(argv)
    cfg = resolve_config(args.config) if args.config else None
    report = inspect_cluster(cfg)
    if args.json:
        print(json.dumps(report, indent=2))
        return 1 if report["errors"] else 0
    return print_report(report, cfg["nodes"] if cfg else ["localhost"])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"RoCE discovery: {error}", file=sys.stderr)
        raise SystemExit(2)
