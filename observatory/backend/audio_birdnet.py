"""Optional, offline BirdNET v2.4 TFLite smoke test, never training truth.

Supply a locally downloaded FP32 model and its matching label file explicitly.
This module does not download models, use location priors, or publish results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .audio_candidates import decode_audio


def predict_windows(interpreter,labels,samples,top_k=5):
    """v2.4 uses float32 3 s/48 kHz input and logits, then sigmoid at sensitivity 1."""
    inputs=interpreter.get_input_details();outputs=interpreter.get_output_details()
    if len(inputs)!=1 or list(inputs[0]['shape'])!=[1,144000] or inputs[0]['dtype']!=np.float32:
        raise ValueError('Expected the BirdNET v2.4 FP32 48 kHz/3-second input contract.')
    if len(outputs)!=1 or int(outputs[0]['shape'][-1])!=len(labels):
        raise ValueError('Model output and label count do not match.')
    if not 1<=top_k<=len(labels):raise ValueError('Invalid top-k count.')
    samples=np.asarray(samples,dtype=np.float32)
    if samples.ndim!=1 or not np.all(np.isfinite(samples)):raise ValueError('Expected finite mono samples.')
    rows=[]
    for offset in range(0,len(samples),144000):
        chunk=samples[offset:offset+144000]
        if len(chunk)<48000:break
        padded=np.zeros((1,144000),dtype=np.float32);padded[0,:len(chunk)]=chunk
        interpreter.set_tensor(inputs[0]['index'],padded)
        tick=time.perf_counter();interpreter.invoke()
        elapsed=time.perf_counter()-tick
        logits=interpreter.get_tensor(outputs[0]['index']).reshape(-1)
        if not np.all(np.isfinite(logits)):raise ValueError('Model returned non-finite scores.')
        scores=1/(1+np.exp(-np.clip(logits,-15,15)))
        top=np.argsort(scores)[-top_k:][::-1]
        rows.append({'start_seconds':offset/48000,'end_seconds':(offset+len(chunk))/48000,
            'padded_seconds':(144000-len(chunk))/48000,'inference_seconds':round(elapsed,4),
            'top_scores':[{'label':labels[int(i)],'score':round(float(scores[i]),6)} for i in top],
            'validated':False,'individual':None})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=3)
    parser.add_argument('--seconds',type=float,default=9)
    parser.add_argument('--source-rate',type=int,default=8000)
    args=parser.parse_args()
    if not 1<=args.limit<=9 or not 1<=args.seconds<=30:parser.error('Smoke-test budget: 1–9 files, 1–30 seconds each.')
    if args.output.exists():parser.error('Use a new output filename.')
    from ai_edge_litert.interpreter import Interpreter
    paths=sorted(p for p in args.input.iterdir() if p.suffix.lower() in {'.mp4','.wav','.m4a','.flac'} and p.is_file())[:args.limit] if args.input.is_dir() else [args.input]
    labels=args.labels.read_text(encoding='utf-8-sig').splitlines()
    tick=time.perf_counter();model=Interpreter(model_path=str(args.model.resolve(strict=True)),num_threads=1);model.allocate_tensors()
    load_seconds=time.perf_counter()-tick
    rows=[]
    for number,path in enumerate(paths,1):
        samples,_,metadata=decode_audio(path,args.seconds,output_rate=48000)
        rows.append({'file_id':f'clip-{number:03d}','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            **metadata,'windows':predict_windows(model,labels,samples)})
    report={'model':'BirdNET v2.4 FP32','model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
        'model_license':'CC BY-NC-SA 4.0','runtime':'ai-edge-litert','threads':1,'model_load_seconds':round(load_seconds,4),
        'model_input_rate':48000,'original_source_rate':args.source_rate,'location_prior':None,
        'interpretation':'Unvalidated sigmoid scores, not calibrated probabilities. No bird identity or singing/call classification.',
        'files':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'files':len(rows),'windows':sum(len(r['windows']) for r in rows),
        'model_load_seconds':report['model_load_seconds'],'inference_seconds':round(sum(w['inference_seconds'] for r in rows for w in r['windows']),4)},indent=2))


if __name__=='__main__':main()
