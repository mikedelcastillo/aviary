"""Offline sound-activity review candidates. No species or identity claims.

python -m backend.audio_candidates --input /local/copied/clips --output /local/review --source-rate 8000
Requires NumPy, ffmpeg and ffprobe. No model downloads or network calls.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
from html import escape
import json
from pathlib import Path
import subprocess
import time
import wave

import numpy as np


@dataclass(frozen=True)
class Settings:
    frame_seconds: float = .04
    hop_seconds: float = .02
    minimum_seconds: float = .08
    merge_gap_seconds: float = .24
    context_seconds: float = .4
    maximum_segment_seconds: float = 6
    maximum_candidates: int = 3
    minimum_dbfs: float = -50
    excess_db: float = 6


def segment_audio(samples, sample_rate, source_rate=None, settings=Settings()):
    """Rank upper-band energy changes/tones, not the probability of a bird."""
    samples=np.asarray(samples,dtype=np.float32)
    if samples.ndim!=1 or not np.all(np.isfinite(samples)):
        raise ValueError('Expected finite mono audio samples.')
    if sample_rate<4000 or source_rate is not None and source_rate<4000:
        raise ValueError('Audio sample rate must be at least 4000 Hz.')
    if settings.maximum_candidates<1 or settings.maximum_segment_seconds<=0:
        raise ValueError('Candidate limits must be positive.')
    duration=len(samples)/sample_rate
    result={'duration_seconds':round(duration,4),'sample_rate':sample_rate,
        'source_sample_rate':source_rate,'source_nyquist_hz':min(sample_rate,source_rate or sample_rate)/2,
        'clipped_sample_fraction':round(float(np.mean(np.abs(samples)>=.999)) if len(samples) else 0,6),
        'segments':[],'interpretation':'Unvalidated sound-activity candidates; not bird, species, behavior, or identity labels.'}
    length=round(settings.frame_seconds*sample_rate);hop=round(settings.hop_seconds*sample_rate)
    if len(samples)<length:return result
    frames=np.lib.stride_tricks.sliding_window_view(samples,length)[::hop]
    frames=frames-np.mean(frames,axis=1,keepdims=True)
    window=np.hanning(length).astype(np.float32)
    power=np.abs(np.fft.rfft(frames*window,axis=1))**2
    freq=np.fft.rfftfreq(length,1/sample_rate)
    upper=min(3500.,.45*min(sample_rate,source_rate or sample_rate))
    band=(freq>=700)&(freq<=upper)
    band_power=power[:,band]
    total=power.sum(axis=1)+1e-20
    fraction=band_power.sum(axis=1)/total
    # Parseval scaling of a one-sided, windowed real FFT.
    energy=2*band_power.sum(axis=1)/(length*np.sum(window**2))
    db=10*np.log10(np.maximum(energy,1e-12))
    floor=float(np.quantile(db,.2))
    flatness=np.exp(np.mean(np.log(band_power+1e-20),axis=1))/(np.mean(band_power,axis=1)+1e-20)
    changes=(db>=max(settings.minimum_dbfs,floor+settings.excess_db))&(fraction>=.12)
    tones=(db>=-38)&(fraction>=.45)&(flatness<=.12)
    active=changes|tones
    result.update({'band_hz':[700,upper],'noise_floor_dbfs':round(floor,2),
        'active_fraction':round(float(np.mean(active)),4)})
    indices=np.flatnonzero(active)
    if not len(indices):return result
    groups=np.split(indices,np.flatnonzero(np.diff(indices)*settings.hop_seconds>settings.merge_gap_seconds)+1)
    candidates=[]
    for group in groups:
        if len(group)*settings.hop_seconds<settings.minimum_seconds:continue
        first,last=int(group[0]),int(group[-1])
        # Bound long busy passages; the candidate budget is per clip.
        width=max(1,round(settings.maximum_segment_seconds/settings.hop_seconds))
        for start in range(first,last+1,width):
            end=min(start+width-1,last)
            hits=group[(group>=start)&(group<=end)]
            if len(hits)*settings.hop_seconds<settings.minimum_seconds:continue
            a=max(0,start*settings.hop_seconds-settings.context_seconds)
            b=min(duration,end*settings.hop_seconds+settings.frame_seconds+settings.context_seconds)
            reasons=[]
            if np.any(changes[hits]):reasons.append('band_energy_increase')
            if np.any(tones[hits]):reasons.append('tonal_band_energy')
            candidates.append({'start_seconds':round(a,3),'end_seconds':round(b,3),
                'rank_score':round(float(np.max(db[hits]-floor)+10*np.mean(fraction[hits])),3),
                'peak_band_dbfs':round(float(np.max(db[hits])),2),'reasons':reasons,
                'review_label':None,'species':None,'individual':None})
    chosen=[]
    for item in sorted(candidates,key=lambda s:s['rank_score'],reverse=True):
        if any(min(item['end_seconds'],c['end_seconds'])>max(item['start_seconds'],c['start_seconds']) for c in chosen):continue
        chosen.append(item)
        if len(chosen)>=settings.maximum_candidates:break
    result['segments']=sorted(chosen,key=lambda s:s['start_seconds'])
    return result


def decode_audio(path, maximum_seconds=120, output_rate=None):
    path=Path(path).resolve(strict=True)
    if not path.is_file():raise ValueError('Input must be a local file.')
    if not 0<maximum_seconds<=600:raise ValueError('Choose 1–600 seconds per copied clip.')
    info=subprocess.run(['ffprobe','-v','error','-select_streams','a:0','-show_entries',
        'stream=sample_rate,channels,codec_name:format=duration','-of','json',str(path)],
        capture_output=True,check=True,timeout=20)
    metadata=json.loads(info.stdout)
    if not metadata.get('streams'):raise ValueError('No audio stream.')
    stream=metadata['streams'][0];native_rate=int(stream['sample_rate'])
    rate=output_rate or min(native_rate,16000)
    if rate<4000 or rate>96000:raise ValueError('Unsupported analysis sample rate.')
    decoded=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(path),'-map','0:a:0','-vn',
        '-t',str(maximum_seconds),'-ac','1','-ar',str(rate),'-f','f32le','pipe:1'],
        capture_output=True,check=True,timeout=max(30,maximum_seconds*2))
    samples=np.frombuffer(decoded.stdout,dtype='<f4').copy()
    return samples,rate,{'container_sample_rate':native_rate,'codec':stream['codec_name'],
        'channels':stream.get('channels'),'container_duration_seconds':float(metadata.get('format',{}).get('duration',len(samples)/rate))}


def write_wav(path,samples,rate):
    with wave.open(str(path),'wb') as audio:
        audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(rate)
        audio.writeframes((np.clip(samples,-1,1)*32767).astype('<i2').tobytes())


def benchmark(inputs, output, source_rate=None, settings=Settings(), maximum_seconds=120):
    """Read completed local copies, create a separate private listening package."""
    inputs=[Path(p).resolve(strict=True) for p in inputs]
    output=Path(output).resolve()
    if any(p.is_relative_to(output) for p in inputs):
        raise ValueError('Keep output separate from the source files.')
    output.mkdir(parents=True,exist_ok=True)
    # Require a fresh run directory so a report cannot mix old and new evidence.
    if any(output.iterdir()):raise ValueError('Use an empty output directory for each review run.')
    started=time.perf_counter();rows=[]
    for number,path in enumerate(inputs,1):
        before=path.stat();tick=time.perf_counter()
        row={'file_id':f'clip-{number:03d}','source_name':path.name}
        try:
            samples,rate,metadata=decode_audio(path,maximum_seconds)
            after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise ValueError('Input changed while decoding; use a completed clip copy.')
            row.update(segment_audio(samples,rate,source_rate,settings));row.update(metadata)
            row['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            row['analysis_seconds']=round(time.perf_counter()-tick,4)
            # Include an independent middle 3 s audit, even when there is no candidate.
            audit_start=max(0,(len(samples)/rate-3)/2)
            audit={'start_seconds':round(audit_start,3),'end_seconds':round(min(len(samples)/rate,audit_start+3),3),'kind':'independent_audit'}
            row['audit']=audit
            for index,item in enumerate([*row['segments'],audit]):
                name=f"{row['file_id']}-{index+1:02d}.wav"
                write_wav(output/name,samples[round(item['start_seconds']*rate):round(item['end_seconds']*rate)],rate)
                item['audio']=name
        except (ValueError,OSError,subprocess.SubprocessError) as error:
            row['error']=type(error).__name__  # ffmpeg errors may contain private paths.
        rows.append(row)
    seconds=sum(r.get('duration_seconds',0) for r in rows)
    summary={'files':len(rows),'decoded_files':sum('error' not in r for r in rows),
        'audio_seconds':round(seconds,3),'candidate_count':sum(len(r.get('segments',[])) for r in rows),
        'candidate_seconds':round(sum(s['end_seconds']-s['start_seconds'] for r in rows for s in r.get('segments',[])),3),
        'analysis_seconds':round(sum(r.get('analysis_seconds',0) for r in rows),4),
        'elapsed_seconds':round(time.perf_counter()-started,4),
        'accuracy':'Not measured. No human ground truth; candidate counts are not bird counts.'}
    report={'version':1,'settings':asdict(settings),'summary':summary,'files':rows}
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    cards=[]
    for row in rows:
        items=[*row.get('segments',[]),*([row['audit']] if 'audit' in row else [])]
        for item in items:
            kind='Independent audit' if item.get('kind') else 'Sound candidate'
            cards.append(f'<article><h2>{escape(row["file_id"])} · {kind}</h2><p>{item["start_seconds"]:.2f}–{item["end_seconds"]:.2f} seconds</p><audio controls preload="none" src="{escape(item["audio"],quote=True)}"></audio><p>Listen and record: bird sound / other sound / uncertain. Singing and identity need separate evidence.</p></article>')
    html='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Local sound review</title><style>body{font:16px system-ui;max-width:760px;margin:24px auto;padding:0 20px;background:#101813;color:#e5eee5}article{border-top:1px solid #506050;padding:16px 0}audio{width:100%}p{line-height:1.6}</style><h1>Local sound review</h1><p>Private audio copies. Candidates are unvalidated; speech and household sounds can appear. Nothing is uploaded. Compare candidates with the independent audit samples.</p>'+''.join(cards)
    (output/'index.html').write_text(html,encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='Directory of completed local copies, or one local media file')
    parser.add_argument('--output',type=Path,required=True,help='New, empty private review directory')
    parser.add_argument('--source-rate',type=int,help='Known original microphone sample rate before transcoding')
    parser.add_argument('--limit',type=int,default=9)
    parser.add_argument('--max-seconds',type=float,default=120)
    args=parser.parse_args()
    if not 1<=args.limit<=100:parser.error('--limit must be 1–100')
    paths=sorted(p for p in args.input.iterdir() if p.suffix.lower() in {'.mp4','.wav','.m4a','.flac'} and p.is_file())[:args.limit] if args.input.is_dir() else [args.input]
    if not paths:parser.error('No local audio/video files found.')
    try:report=benchmark(paths,args.output,args.source_rate,maximum_seconds=args.max_seconds)
    except (ValueError,OSError) as error:parser.error(str(error))
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':main()
