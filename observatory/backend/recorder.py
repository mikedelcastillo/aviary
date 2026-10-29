"""One decode per camera: steady video, recordings, and latest inference JPEG."""
import json
import argparse
import os
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from .store import configuration
from .recorder_health import RecorderLimits, RecorderWatchdog, child_memory, playlist_version, stop_child
from .recorder_hls import HLSPublisher


def command_for(config, camera, encoder_name, source=None, input_options=None, wallclock=True, publisher=None):
    root = Path(config['runtime'])
    hls = root/'hls'/camera['id']; hls.mkdir(parents=True, exist_ok=True)
    recordings = root/'recordings'/camera['id']; recordings.mkdir(parents=True, exist_ok=True)
    capture = root/'capture'; capture.mkdir(parents=True, exist_ok=True)
    encoder = ['-c:v','h264_nvenc','-preset','p4','-tune','ll','-rc','vbr','-cq','24','-b:v','1500k','-maxrate','2500k','-bufsize','3M']
    if encoder_name == 'libx264':
        encoder=['-c:v','libx264','-preset','veryfast','-crf','24','-threads','2']
    # Tee shares one encoded packet stream between HLS and clip segments.
    sequence = f'start_number={publisher.start}' if publisher else 'hls_start_number_source=epoch'
    playlist = publisher.incoming if publisher else hls/'live.m3u8'
    flags = 'independent_segments+program_date_time+temp_file+omit_endlist'
    if publisher is None:
        flags += '+delete_segments'
    outputs=(f'[f=hls:hls_time=2:hls_list_size=10:hls_delete_threshold=8:{sequence}:'
        f'hls_flags={flags}:hls_segment_filename={hls}/s%010d.ts]{playlist}'
        f'|[f=segment:segment_time=20:reset_timestamps=1:strftime=1:segment_format=mp4:segment_format_options=movflags=+faststart]{recordings}/%Y%m%dT%H%M%SZ.mp4')
    # Bound stream discovery and the queue waiting for a delayed optional audio stream.
    # FFmpeg 6.1 has no filter_buffered_frames option; memory supervision remains necessary.
    return ['ffmpeg','-y','-nostdin','-loglevel','error','-progress','pipe:1','-stats_period','1','-filter_threads','2',
        '-fflags','+genpts+discardcorrupt','-probesize','1048576','-analyzeduration','2000000',
        '-rtbufsize','4194304',
        *(input_options if input_options is not None else ['-rtsp_transport','tcp','-timeout','10000000']),
        *(['-use_wallclock_as_timestamps','1'] if wallclock else []),'-threads','2',
        '-i',source or config['relay_rtsp']+'/'+camera['id'],
        '-map','0:v:0','-map','0:a:0?','-vf','fps=15,scale=1280:720,setpts=N/(15*TB)',*encoder,
        '-c:a','aac','-b:a','64k','-ar','16000','-ac','1','-af','aresample=async=1:first_pts=0,asetpts=N/SR/TB',
        '-max_muxing_queue_size','128','-muxing_queue_data_threshold','1048576',
        '-pix_fmt','yuv420p','-g','30','-bf','0','-profile:v','main','-flags','+global_header','-f','tee',outputs,
        '-map','0:v:0','-an','-vf','fps=1,scale=1280:720','-c:v','mjpeg','-q:v','4','-threads','1',
        '-f','image2','-update','1','-atomic_writing','1',str(capture/(camera['id']+'.jpg'))]


def supervise(proc, playlist, limits, on_sample, stop=None, publisher=None):
    """Watch fresh output and this process's memory, not a sibling's cached playlist."""
    started = time.monotonic()
    watchdog = RecorderWatchdog(limits, started, playlist_version(playlist))
    while proc.poll() is None:
        if publisher is not None:
            publisher.publish()
        now = time.monotonic()
        memory = child_memory(proc.pid)
        reason = watchdog.observe(now, playlist_version(playlist), memory)
        if stop is not None and stop.is_set():
            reason = 'shutdown'
        on_sample({'child_pid': proc.pid, 'memory': memory, 'output_started': watchdog.output_started,
                   'output_idle_seconds': round(now - watchdog.last_output, 3), 'stop_reason': reason})
        if reason:
            stop_child(proc, limits, emergency=reason == 'memory-emergency')
            return reason
        if stop is None:
            time.sleep(limits.poll_seconds)
        else:
            stop.wait(limits.poll_seconds)
    return 'process-exit'


def record(config, camera, stop=None):
    stop = stop or threading.Event()
    root = Path(config['runtime'])
    playlist = root/'hls'/camera['id']/'live.m3u8'
    heartbeat = root/'recorder'; heartbeat.mkdir(parents=True, exist_ok=True)
    limits = RecorderLimits.from_config(config)
    failures = 0
    software_until = 0
    while not stop.is_set():
        started = time.time()
        started_mono = time.monotonic()
        encoder = 'libx264' if config.get('video_encoder','h264_nvenc') == 'libx264' or failures >= 3 or started_mono < software_until else 'h264_nvenc'
        publisher = HLSPublisher(playlist)
        command = command_for(config, camera, encoder, publisher=publisher)
        # -y is necessary even for image2's update mode: an existing latest JPEG
        # otherwise makes ffmpeg exit at startup after a service restart.
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True, env={**os.environ,'TZ':'UTC'})
        errors = deque(maxlen=6)
        progress = {}
        def read_errors():
            for line in proc.stderr:
                errors.append(line.decode(errors='replace').strip()[:400])
        def read_progress():
            for line in proc.stdout:
                key, _, value = line.decode(errors='replace').strip().partition('=')
                if key in {'frame', 'out_time_us', 'speed', 'progress'}:
                    progress[key] = value[:80]
        readers = [threading.Thread(target=read_errors, daemon=True), threading.Thread(target=read_progress, daemon=True)]
        for thread in readers: thread.start()
        def publish(sample):
            now=time.time()
            try: age=now-playlist.stat().st_mtime
            except FileNotFoundError: age=None
            state={'timestamp':now,'started':started,'playlist_age':age,'restarts':failures,
                   'encoder':encoder,'errors':list(errors)[-3:], 'progress':dict(progress), **sample}
            target=heartbeat/(camera['id']+'.json');temporary=target.with_suffix('.tmp')
            temporary.write_text(json.dumps(state));temporary.replace(target)
        try:
            reason = supervise(proc, playlist, limits, publish, stop, publisher)
        except (OSError, ValueError):
            reason = 'playlist-error'
            errors.append('HLS publication failed; recorder child will be stopped')
        finally:
            if proc.poll() is None: stop_child(proc, limits)
            try:
                publisher.publish()  # Include any finalized segment after SIGINT.
            except (OSError, ValueError):
                errors.append('Final HLS publication failed')
            for thread in readers: thread.join(timeout=1)
            proc.stdout.close(); proc.stderr.close()
        code = proc.wait()
        if time.monotonic()-started_mono>120: failures=0
        failures += 1
        if encoder == 'h264_nvenc' and reason.startswith('memory-'):
            software_until = time.monotonic() + 600
        print(json.dumps({'camera':camera['id'],'exit':code,'lifetime_seconds':round(time.monotonic()-started_mono),
                          'restarts':failures,'reason':reason,'errors':list(errors)[-3:]}),flush=True)
        stop.wait(3)


def retain(config, stop=None):
    """Only remove generated rolling segments from this new application's directory."""
    folder = (Path(config['runtime'])/'recordings').resolve()
    budget = config.get('recording_budget_gb',8)*1024**3
    stop = stop or threading.Event()
    while not stop.is_set():
        files = sorted((p for p in folder.glob('*/*.mp4') if p.is_file() and p.resolve().is_relative_to(folder)),key=lambda p:p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        for path in files:
            if total <= budget: break
            stat = path.stat()
            if time.time()-stat.st_mtime < 120: continue  # never unlink an active/incomplete segment
            total -= stat.st_size
            path.unlink()
        stop.wait(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', help='Run one configured camera in an independently limited service')
    parser.add_argument('--retention-only', action='store_true')
    args = parser.parse_args()
    config=configuration()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    if args.camera and args.retention_only: parser.error('Choose a camera or retention-only')
    if args.camera:
        camera = next((camera for camera in config['cameras'] if camera['id'] == args.camera), None)
        if camera is None: parser.error('Unknown configured camera')
        record(config, camera, stop)
    else:
        threads = []
        if not args.retention_only:
            for camera in config['cameras']:
                thread = threading.Thread(target=record,args=(config,camera,stop),daemon=True)
                thread.start(); threads.append(thread)
        try:
            retain(config, stop)
        finally:
            stop.set()
            for thread in threads: thread.join(timeout=8)
