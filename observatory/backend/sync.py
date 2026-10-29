"""Read source journals over SSH; write/import copies on this host only."""
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from .store import configuration, import_journal


def run():
    config = configuration()
    destination = Path(config['runtime']) / 'journals'
    destination.mkdir(parents=True, exist_ok=True)
    while True:
        source = config.get('journal_source')
        if source:
            today = datetime.now(ZoneInfo(config['timezone'])).date()
            for day in [today-timedelta(days=1), today]:
                name = day.isoformat()+'.jsonl'
                result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', source['host'],
                    'cat -- ' + shlex.quote(source['root'] + '/' + name)], capture_output=True, timeout=30)
                if result.returncode == 0:
                    path = destination/name
                    path.write_bytes(result.stdout)
                    if not config.get('archive_binary'):
                        print(json.dumps({'date': str(day), 'imported': import_journal(config, path)}), flush=True)
            if config.get('archive_binary'):
                result = subprocess.run([config['archive_binary'], os.environ['AVIARY_HOME_CONFIG'], str(destination)], capture_output=True, text=True, timeout=120)
                print(result.stdout.strip() if result.returncode == 0 else 'Archive import failed', flush=True)
        time.sleep(120)


if __name__ == '__main__':
    run()
