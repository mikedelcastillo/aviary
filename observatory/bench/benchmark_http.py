"""Compare native and Python HTTP paths against one isolated archive snapshot.

Run on a development host, never against a production listening port. Outputs
aggregate timings and memory only; private records remain in the temporary copy.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import statistics
import subprocess
import tempfile
import time

import httpx


def summary(samples):
    ordered = sorted(samples)
    return {"requests": len(samples), "median_ms": round(statistics.median(samples), 3),
            "p95_ms": round(ordered[int((len(ordered)-1)*.95)], 3)}


async def exercise(url):
    headers = {"Cookie": "aviary_session=fictional-benchmark-session"}
    async with httpx.AsyncClient(base_url=url, headers=headers, timeout=30) as client:
        for _ in range(10):
            (await client.get('/api/events?limit=24')).raise_for_status()
        sample = (await client.get('/api/events?limit=24')).json()
        signature = hashlib.sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest()
        sequential = []
        for _ in range(200):
            start = time.perf_counter()
            (await client.get('/api/events?limit=24')).raise_for_status()
            sequential.append((time.perf_counter()-start)*1000)
        semaphore = asyncio.Semaphore(16)

        async def request():
            async with semaphore:
                start = time.perf_counter()
                (await client.get('/api/events?limit=24')).raise_for_status()
                return (time.perf_counter()-start)*1000

        start = time.perf_counter()
        concurrent = await asyncio.gather(*(request() for _ in range(800)))
        throughput = 800/(time.perf_counter()-start)
        media = []
        for _ in range(80):
            start = time.perf_counter()
            response = await client.get('/api/clips/benchmark/video', headers={'Range': 'bytes=0-1048575'})
            assert response.status_code == 206 and len(response.content) == 1048576
            media.append((time.perf_counter()-start)*1000)
        return {"payload_sha256": signature, "events_sequential": summary(sequential),
                "events_concurrency_16": {**summary(concurrent), "requests_per_second": round(throughput, 1)},
                "range_1_mib": summary(media)}


def run(args):
    source = json.loads(Path(args.config).read_text())
    with tempfile.TemporaryDirectory(prefix='aviary-http-benchmark-') as folder:
        root = Path(folder)
        with sqlite3.connect(Path(source['runtime'])/'aviary.sqlite') as original:
            with sqlite3.connect(root/'aviary.sqlite') as copy:
                original.backup(copy)
        config = {**source, 'runtime': str(root), 'cameras': [], 'birds': []}
        config_path = root/'config.json'
        config_path.write_text(json.dumps(config))
        config_path.chmod(0o600)
        (root/'sample.mp4').write_bytes(bytes(range(256))*8192)
        with sqlite3.connect(root/'aviary.sqlite') as db:
            db.execute('INSERT OR REPLACE INTO sessions VALUES(?,?)',
                       (hashlib.sha256(b'fictional-benchmark-session').hexdigest(), time.time()+3600))
            db.execute("INSERT OR REPLACE INTO clips(id,time,camera,path,duration) VALUES('benchmark','2026-01-01','example','sample.mp4',20)")
            count = db.execute('SELECT count(*) FROM events').fetchone()[0]
        output = {'archive_records': count, 'method': 'same isolated SQLite snapshot; warm loopback HTTP; 16 in-flight requests; no TLS'}
        for name in ['python', 'rust']:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            command = [args.python, '-m', 'uvicorn', 'backend.app:app', '--host', '127.0.0.1', '--port', str(port), '--no-access-log'] if name == 'python' else [args.rust_bin]
            process = subprocess.Popen(command, env={**os.environ, 'AVIARY_HOME_CONFIG': str(config_path),
                'AVIARY_LISTEN': f'127.0.0.1:{port}', 'AVIARY_WEB_ROOT': args.web,
                'AVIARY_ADAPTER_URL': 'http://127.0.0.1:1'}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                url = f'http://127.0.0.1:{port}'
                for _ in range(100):
                    try:
                        if httpx.get(url+'/api/health').status_code == 200:
                            break
                    except httpx.HTTPError:
                        time.sleep(.05)
                else:
                    raise RuntimeError(f'{name} failed to start')
                result = asyncio.run(exercise(url))
                status = dict(line.split(':', 1) for line in Path(f'/proc/{process.pid}/status').read_text().splitlines())
                result['rss_kib'] = int(status['VmRSS'].split()[0])
                result['peak_rss_kib'] = int(status['VmHWM'].split()[0])
                output[name] = result
            finally:
                process.terminate()
                try:
                    process.wait(10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        output['same_payload'] = output['python']['payload_sha256'] == output['rust']['payload_sha256']
        assert output['same_payload'], 'Native and Python payloads differ'
        Path(args.output).write_text(json.dumps(output, indent=2)+'\n')
        print(json.dumps(output, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ['config', 'rust-bin', 'python', 'web', 'output']:
        parser.add_argument('--'+key, required=True)
    run(parser.parse_args())
