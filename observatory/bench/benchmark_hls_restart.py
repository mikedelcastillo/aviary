#!/usr/bin/env python3
"""Three real CPU encoders and existing hls.js survive local offline faults.

All input is an existing local copy. No cameras, relay or GPU encoders are used.
"""
import argparse
import functools
import http.server
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.recorder import command_for
from backend.recorder_health import RecorderLimits, stop_child
from backend.recorder_hls import HLSPublisher, read_playlist


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hls-js',type=Path,required=True)
    parser.add_argument('--browser-script',type=Path,required=True)
    args=parser.parse_args()
    if not args.fixture.is_file() or not args.hls_js.is_file():parser.error('Existing local fixture and hls.js are required')
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()):parser.error('Use an empty isolated output directory')
    output=args.output.resolve();fixture=output/'fixture-av.mp4'
    subprocess.run(['ffmpeg','-y','-nostdin','-v','error','-i',str(args.fixture.resolve()),'-f','lavfi','-i','sine=frequency=440:sample_rate=16000',
                    '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','64k','-t','12','-shortest',str(fixture)],check=True)
    shutil.copyfile(args.hls_js,output/'hls.js')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Offline HLS restart test</title><script src="/hls.js"></script>')
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self,*_):pass
        def end_headers(self):
            self.send_header('Cache-Control','no-store');super().end_headers()
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(QuietHandler,directory=str(output)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    processes=[];workers=[];logs=[];generations={};active={};events=[];manifest_checks=[]
    known={};stop=threading.Event();failure=[];lock=threading.Lock()
    began=time.monotonic()
    def launch(camera):
        publisher=HLSPublisher(output/'hls'/camera/'live.m3u8')
        generation=generations.get(camera,0);generations[camera]=generation+1
        command=command_for({'runtime':str(output),'relay_rtsp':''},{'id':camera},'libx264',str(fixture),['-re','-stream_loop','-1'],wallclock=False,publisher=publisher)
        log=(output/f'{camera}-{generation}.stderr').open('wb');logs.append(log)
        process=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=log,start_new_session=True,env={**os.environ,'TZ':'UTC'})
        processes.append(process);active[camera]=(process,publisher)
        def publish():
            try:
                while process.poll() is None and not stop.is_set():
                    if publisher.publish():check(camera,publisher)
                    time.sleep(.1)
                if publisher.publish():check(camera,publisher)
            except Exception as error:failure.append(repr(error))
        thread=threading.Thread(target=publish);thread.start();workers.append(thread)
        events.append({'at':round(time.monotonic()-began,3),'camera':camera,'event':'start','generation':generation})
        return process
    def check(camera,publisher):
        rows,target=read_playlist(publisher.playlist)
        with lock:
            for row in rows:
                key=(camera,row.uri)
                assert known.setdefault(key,row)==row,'An advertised segment identity/PDT/discontinuity changed'
            assert len(rows)<=10
            manifest_checks.append({'at':round(time.monotonic()-began,3),'camera':camera,'first':rows[0].sequence,'last':rows[-1].sequence,'cc':rows[0].discontinuity,'last_cc':rows[-1].discontinuity})
    def halt(camera,emergency):
        process,publisher=active[camera]
        stop_child(process,RecorderLimits(shutdown_seconds=2),emergency=emergency)
        for thread in workers:thread.join(timeout=.2)
        events.append({'at':round(time.monotonic()-began,3),'camera':camera,'event':'kill' if emergency else 'graceful-stop'})
    def wait(seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if failure:raise RuntimeError(failure[0])
            time.sleep(.1)
    browser=None
    try:
        for camera in ['fixture-1','fixture-2','fixture-3']:launch(camera)
        deadline=time.monotonic()+25
        while not all(len(read_playlist(p)[0])>=3 if p.exists() else False for p in [output/'hls'/c/'live.m3u8' for c in active]):
            if time.monotonic()>deadline:raise RuntimeError('Offline fixture startup timed out')
            wait(.2)
        browser=subprocess.Popen(['node',str(args.browser_script.resolve()),f'http://127.0.0.1:{server.server_port}',str(output)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        deadline=time.monotonic()+25
        while not (output/'browser-ready').exists():
            if browser.poll() is not None:raise RuntimeError(browser.communicate()[1])
            if time.monotonic()>deadline:raise RuntimeError('Browser startup timed out')
            wait(.2)
        wait(10);halt('fixture-1',False);wait(1);launch('fixture-1')
        wait(10);halt('fixture-2',True);wait(2);launch('fixture-2')
        wait(10);halt('fixture-3',True);wait(23);launch('fixture-3')
        wait(28)
        (output/'finish').write_text('done')
        stdout,stderr=browser.communicate(timeout=15)
        if browser.returncode:raise RuntimeError(stderr)
        result=json.loads((output/'browser-result.json').read_text())
        checks=[]
        for player in result['players']:
            samples=[s for s in result['samples'] if s['id']==player['id']]
            assert player['frames']>500 and player['time']>30
            assert not player['paused'] and player['ready']>=2
            assert not any(error['fatal'] for error in player['errors']),player['errors']
            assert player['ended']==0
            assert player['fragments'][-1]['cc']==1
            assert player['time']-samples[-20]['time']>3,'Playback did not advance after recovery'
            assert player['latency']<12,'Player remained far behind the live edge'
            checks.append({'id':player['id'],'frames':player['frames'],'final_latency_seconds':round(player['latency'],3),
                           'fatal_errors':0,'ended':0,'final_discontinuity':1,'nonfatal_errors':len(player['errors'])})
        summary={'offline_only':True,'encoder':'libx264','audio':'synthetic AAC 16kHz','hls_version':result['version'],
                 'player_replacements':0,'manifest_invariants_checked':len(manifest_checks),'players':checks,'events':events}
        (output/'result.json').write_text(json.dumps(summary,indent=2))
        (output/'manifest-checks.json').write_text(json.dumps(manifest_checks,indent=2))
        print(json.dumps(summary,indent=2))
    finally:
        stop.set()
        for process in processes:
            if process.poll() is None:stop_child(process,RecorderLimits(shutdown_seconds=1))
        for thread in workers:thread.join(timeout=2)
        if browser and browser.poll() is None:browser.terminate();browser.wait(timeout=5)
        for log in logs:log.close()
        server.shutdown();server.server_close()


if __name__=='__main__':main()
