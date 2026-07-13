"""MAC normalization and ARP-table lookups for the camera watchlist."""

from __future__ import annotations

import pytest

from lib.netid import format_mac, mac_for_ip, normalize_mac


# --- normalize / format -----------------------------------------------------


def test_normalize_accepts_common_spellings() -> None:
    canonical = "cc:ba:bd:9a:ef:51"
    assert normalize_mac("CC-BA-BD-9A-EF-51") == canonical
    assert normalize_mac("cc:ba:bd:9a:ef:51") == canonical
    assert normalize_mac("CCBA.BD9A.EF51") == canonical
    assert normalize_mac("ccbabd9aef51") == canonical


def test_normalize_rejects_junk() -> None:
    for raw in ("", "not-a-mac", "cc:ba:bd:9a:ef", "cc:ba:bd:9a:ef:51:22", None):
        with pytest.raises(ValueError):
            normalize_mac(raw)  # type: ignore[arg-type]


def test_format_mac_is_uppercase_dashed() -> None:
    assert format_mac("cc:ba:bd:9a:ef:51") == "CC-BA-BD-9A-EF-51"


# --- ARP table lookup ---------------------------------------------------------

_ARP_TABLE = """IP address       HW type     Flags       HW address            Mask     Device
192.168.1.64     0x1         0x2         3c:52:a1:0b:22:9e     *        eno1
192.168.1.70     0x1         0x0         00:00:00:00:00:00     *        eno1
192.168.1.71     0x1         0x2         ac:a7:f1:34:3f:8c     *        wlp5s0
"""


def test_mac_for_ip_reads_complete_entry(tmp_path) -> None:
    arp = tmp_path / "arp"
    arp.write_text(_ARP_TABLE)
    assert mac_for_ip("192.168.1.64", arp_path=arp) == "3c:52:a1:0b:22:9e"
    assert mac_for_ip("192.168.1.71", arp_path=arp) == "ac:a7:f1:34:3f:8c"


def test_mac_for_ip_rejects_incomplete_entry(tmp_path) -> None:
    # Flags 0x0 = resolution in flight/failed; the zero MAC must never be
    # treated as a camera identity.
    arp = tmp_path / "arp"
    arp.write_text(_ARP_TABLE)
    assert mac_for_ip("192.168.1.70", arp_path=arp) is None


def test_mac_for_ip_unknown_host_and_missing_table(tmp_path) -> None:
    arp = tmp_path / "arp"
    arp.write_text(_ARP_TABLE)
    assert mac_for_ip("192.168.1.99", arp_path=arp) is None
    assert mac_for_ip("192.168.1.64", arp_path=tmp_path / "absent") is None
