import socket
import pytest
from backend.discovery import advertisement, lan_address


def test_discovery_stays_on_private_lan():
    assert lan_address('192.168.42.20') == '192.168.42.20'
    for address in ['0.0.0.0','127.0.0.1','8.8.8.8','100.64.0.1','169.254.3.4','::1']:
        with pytest.raises(ValueError): lan_address(address)


def test_advertised_identity_has_no_account_or_credentials():
    info = advertisement('192.168.42.20', 8740, 'aviary-fixture')
    assert info.type == '_aviary._tcp.local.'
    assert info.port == 8740
    assert info.addresses == [socket.inet_aton('192.168.42.20')]
    assert info.properties == {b'api':b'1', b'path':b'/', b'scheme':b'http'}
    with pytest.raises(ValueError): advertisement('192.168.42.20', 0, 'aviary-fixture')
