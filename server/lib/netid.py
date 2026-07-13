"""Network identity helpers: resolve a LAN host's MAC address.

Cameras are watched by MAC (stable hardware identity) while streams connect by
IP (whatever DHCP handed out). The kernel already learns every neighbour's MAC
as a side effect of talking to it — the discovery sweep's RTSP probe forces an
ARP resolution for each live host — so this module just reads the ARP table
back. Pure stdlib, matching :mod:`lib.discovery`'s no-dependency philosophy.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

LOGGER = logging.getLogger("lib.netid")

ARP_TABLE_PATH = Path("/proc/net/arp")

# ARP flag bit: entry is COMPLETE (the MAC is resolved and valid). Entries
# without it show 00:00:00:00:00:00 while resolution is in flight or failed.
_ATF_COM = 0x2

_INVALID_MACS = {"00:00:00:00:00:00", "ff:ff:ff:ff:ff:ff"}


def normalize_mac(raw: str) -> str:
    """Canonical lowercase-colon MAC (``aa:bb:cc:dd:ee:ff``) from any common
    spelling — colons, dashes (``AA-BB-...``), Cisco dots, or bare hex.

    Raises ValueError for anything that isn't 12 hex digits; storage and
    comparison always use this form so user input can't miss a match over
    formatting.
    """
    digits = re.sub(r"[^0-9a-fA-F]", "", raw or "")
    if len(digits) != 12:
        raise ValueError(f"Not a MAC address: {raw!r}")
    return ":".join(digits[i : i + 2] for i in range(0, 12, 2)).lower()


def format_mac(mac: str) -> str:
    """Display form (``AA-BB-CC-DD-EE-FF``) of a normalized MAC."""
    return mac.replace(":", "-").upper()


def mac_for_ip(ip: str, *, arp_path: Path = ARP_TABLE_PATH) -> str | None:
    """The MAC last seen for ``ip`` in the kernel ARP table, or None.

    Only COMPLETE entries count — an in-flight or failed resolution reports the
    all-zero placeholder, which must never be stored as a camera identity. The
    caller is expected to have talked to the host recently (the discovery probe
    does), which is what puts the entry in the table in the first place.
    """
    try:
        lines = arp_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        LOGGER.warning("Could not read ARP table at %s", arp_path)
        return None
    for line in lines[1:]:  # first line is the column header
        fields = line.split()
        if len(fields) < 4 or fields[0] != ip:
            continue
        try:
            flags = int(fields[2], 16)
            mac = normalize_mac(fields[3])
        except ValueError:
            continue
        if flags & _ATF_COM and mac not in _INVALID_MACS:
            return mac
    return None
