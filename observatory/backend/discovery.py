"""Advertise Aviary on the local LAN so TV clients need no IP/port entry.

No credentials, roster, host username or camera data are advertised. This is
service discovery, not authentication: clients still verify the API and unlock.
"""
import ipaddress
import json
import signal
import socket
import subprocess
import threading
import urllib.request
import uuid

from .store import configuration


def lan_address(value):
    address = ipaddress.ip_address(value)
    if address.version != 4 or not address.is_private or address.is_loopback or address.is_link_local or address.is_unspecified:
        raise ValueError('Discovery requires a private LAN IPv4 address')
    return str(address)


def local_address(config):
    supplied = config.get('discovery', {}).get('address')
    if supplied:
        return lan_address(supplied)
    route = json.loads(subprocess.check_output(['ip', '-j', 'route', 'get', '224.0.0.251'], timeout=3))
    return lan_address(route[0].get('prefsrc') or route[0]['src'])


def advertisement(address, port, host):
    from zeroconf import ServiceInfo
    if not 1 <= port <= 65535:
        raise ValueError('Invalid discovery port')
    return ServiceInfo('_aviary._tcp.local.', 'Aviary._aviary._tcp.local.',
                       addresses=[socket.inet_aton(lan_address(address))], port=port,
                       properties={'api':'1', 'path':'/', 'scheme':'http'},
                       server=host+'.local.', host_ttl=30, other_ttl=30)


def run(config):
    from zeroconf import IPVersion, Zeroconf
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    port = int(config.get('discovery', {}).get('port', 8740))
    host = 'aviary-'+uuid.uuid4().hex[:12]
    current, zeroconf, info = None, None, None
    try:
        while not stop.is_set():
            try:
                address = local_address(config)
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/discovery', timeout=3) as response:
                    identity = json.load(response)
                if identity != {'service':'aviary-observatory', 'api':1}:
                    raise ValueError('Aviary API unavailable')
                if address != current:
                    if zeroconf:
                        zeroconf.unregister_service(info); zeroconf.close()
                    zeroconf = Zeroconf(interfaces=[address], ip_version=IPVersion.V4Only)
                    info = advertisement(address, port, host)
                    zeroconf.register_service(info, allow_name_change=True)
                    current = address
                    print('Aviary discovery ready on the selected LAN interface', flush=True)
            except Exception as error:
                if zeroconf:
                    zeroconf.unregister_service(info); zeroconf.close()
                    zeroconf, info, current = None, None, None
                print('Discovery waiting: '+type(error).__name__, flush=True)
            stop.wait(15)
    finally:
        if zeroconf:
            zeroconf.unregister_service(info); zeroconf.close()


if __name__ == '__main__':
    run(configuration())
