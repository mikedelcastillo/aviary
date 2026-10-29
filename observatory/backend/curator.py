"""Recordings -> visual evidence -> conservative local-video curation.

Every decision is retained for review. Unclear/static/person clips are never
automatically promoted. No names, health or consumption claims are generated.
"""
import base64
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
import cv2
import httpx
import numpy as np
import torch
from ultralytics import YOLO
from .store import configuration,database,initialize
from .learning_sampling import offer

SCHEMA={'type':'object','properties':{
    'behavior':{'type':'string','enum':['flight','climbing','preening','interaction','at_feeder','at_water','resting','unclear']},
    'caption':{'type':'string'},'worth_watching':{'type':'boolean'},'confidence':{'type':'number'},
    'visible_evidence':{'type':'array','items':{'type':'string'}},'uncertainty':{'type':'string'}},
    'required':['behavior','caption','worth_watching','confidence','visible_evidence','uncertainty']}
SYSTEM='''Judge this short bird video from its ordered sampled frames. Be strict and literal.
Images are in time order, not separate events. Describe only visible action or change.
No bird names, gender, feelings, motivations, health judgments, eating or drinking claims.
A bird next to a bowl is not evidence of consumption. Use at_feeder/at_water only if the vessel is clearly identifiable; otherwise unclear.
Skip static perched/sleeping birds, empty cages, camera movement, blur, obstructed or uncertain behavior.
An interesting clip needs clearly visible flight, climbing, preening or interaction over the sequence.
One short plain caption, no story, puns, emojis, hype or invented details. State uncertainty explicitly.
Return the requested JSON schema. Visible_evidence must describe concrete differences between frame 1, 2 and 3.'''


def frames_at(path,duration):
    capture=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
    images=[]
    try:
        for fraction in [.15,.5,.85]:
            capture.set(cv2.CAP_PROP_POS_MSEC,duration*fraction*1000)
            ok,frame=capture.read()
            if not ok: return []
            images.append(frame)
    finally: capture.release()
    return images


def encode(frame):
    height,width=frame.shape[:2]
    image=cv2.resize(frame,(min(768,width),round(height*min(768,width)/width)))
    return base64.b64encode(cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,80])[1]).decode()


def judge(config,images):
    started=time.monotonic()
    with httpx.Client(timeout=180) as client:
        response=client.post(config['ollama_url']+'/api/chat',json={
            'model':config.get('curator_model','qwen3.5:4b'),'think':False,'stream':False,'format':SCHEMA,'keep_alive':'30s',
            'options':{'temperature':0,'num_ctx':4096,'num_predict':350,'num_thread':6},
            'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':'Frames 1, 2 and 3, at 15%, 50% and 85% of the clip. What actually changes?', 'images':[encode(f) for f in images]}]})
        response.raise_for_status()
        data=json.loads(response.json()['message']['content'])
    data['model']=config.get('curator_model','qwen3.5:4b')
    data['latency_seconds']=round(time.monotonic()-started,3)
    return data


def verdict(analysis,roster):
    """Conservative publication gate; model confidence alone is not calibrated accuracy."""
    caption=str(analysis.get('caption',''))
    forbidden=['healthy','happy','sad','hungry','thirsty','eating','drinking','ate ','drank ','enjoy','love','friend','playful']
    if any(word in caption.lower() for word in forbidden): return False
    import re
    if any(re.search(r'\b'+re.escape(b)+r'\b',caption,re.I) for b in roster): return False
    return (analysis.get('worth_watching') is True and analysis.get('behavior') in ['flight','climbing','preening','interaction']
            and isinstance(analysis.get('confidence'),(int,float)) and analysis['confidence']>=.9
            and len(analysis.get('visible_evidence',[]))>=2 and 5<len(caption)<220)


def initialize_models(config):
    torch.set_num_threads(3);cv2.setNumThreads(1)
    return YOLO(config['person_model']),YOLO(config['bird_model'])


def run():
    config=configuration();initialize(config)
    root=Path(config['runtime']);(root/'clips').mkdir(exist_ok=True);(root/'posters').mkdir(exist_ok=True)
    person,bird_model=initialize_models(config)
    last_model=0
    while True:
        try:
            # A separate lightweight catalog process indexes files every ten
            # seconds, including while this process waits for a model response.
            with database(config) as db:
                candidates=db.execute("SELECT * FROM clips WHERE status='recorded' ORDER BY time DESC LIMIT 9").fetchall()
            for row in candidates:
                path=root/row['path']
                if not path.exists():continue
                images=frames_at(path,row['duration'])
                if len(images)!=3:continue
                poster=root/'posters'/(row['id']+'.jpg')
                cv2.imwrite(str(poster),cv2.resize(images[1],(640,360)),[cv2.IMWRITE_JPEG_QUALITY,78])
                status='quiet';analysis={};caption='';behavior='';score=None
                flags=[len(person.predict(f,imgsz=640,classes=[0],conf=.2,device='cpu',verbose=False)[0].boxes)>0 for f in images]
                if any(flags):status='private';analysis={'reason':'Person detected in a sampled frame'}
                else:
                    results=[bird_model.predict(f,imgsz=960,conf=.25,device='cpu',verbose=False)[0] for f in images]
                    # The offer happens even when there are ZERO bird detections.
                    # These are proposals, never authoritative training labels.
                    h,w=images[1].shape[:2]
                    suggestions=[]
                    for i,box in enumerate(results[1].boxes):
                        x1,y1,x2,y2=box.xyxy[0].tolist()
                        suggestions.append({'id':f'proposal-{i}','cx':(x1+x2)/(2*w),'cy':(y1+y2)/(2*h),
                            'w':(x2-x1)/w,'h':(y2-y1)/h,'label':None,'confidence':float(box.conf[0])})
                    offer(config,row,images[1],suggestions)
                    if any(len(r.boxes)>0 for r in results):
                        a=cv2.resize(cv2.cvtColor(images[0],cv2.COLOR_BGR2GRAY),(160,90))
                        b=cv2.resize(cv2.cvtColor(images[-1],cv2.COLOR_BGR2GRAY),(160,90))
                        movement=float(np.mean(cv2.absdiff(a,b)>25))
                        analysis={'motion_fraction':round(movement,5),'bird_counts':[len(r.boxes) for r in results]}
                        if movement>.25:analysis['reason']='Camera or scene moved';status='review'
                        elif movement>.001 and time.monotonic()-last_model>config.get('curation_interval_seconds',180):
                            try:
                                generated=judge(config,images)
                                analysis.update(generated);last_model=time.monotonic()
                                caption=str(generated.get('caption',''))[:220];behavior=generated.get('behavior','unclear');score=generated.get('confidence')
                                # Automatic publication remains opt-in until a real clip judging evaluation passes.
                                if verdict(generated,[b['id'] for b in config['birds']]):
                                    status='approved' if config.get('curation_auto_publish',False) else 'review'
                                else:status='quiet'
                            except Exception as error:analysis['error']=type(error).__name__;status='review';last_model=time.monotonic()
                        elif movement>.001:
                            status='candidate'
                with database(config) as db:
                    db.execute('UPDATE clips SET poster=?,status=?,caption=?,behavior=?,score=?,analysis=? WHERE id=?',
                        ('posters/'+poster.name,status,caption,behavior,score,json.dumps(analysis),row['id']))
                if status in ('approved','review') and caption:
                    target=root/'clips'/(row['id']+'.mp4');shutil.copy2(path,target)
                    with database(config) as db:db.execute('UPDATE clips SET path=? WHERE id=?',('clips/'+target.name,row['id']))
                print(json.dumps({'clip':row['id'],'status':status,'behavior':behavior}),flush=True)
            # Retry motion candidates at a bounded rate; never build an unbounded inference queue.
            if time.monotonic()-last_model>config.get('curation_interval_seconds',180):
                with database(config) as db:
                    db.execute("UPDATE clips SET status='recorded' WHERE id IN(SELECT id FROM clips WHERE status='candidate' ORDER BY time DESC LIMIT 3)")
        except Exception as error:print(json.dumps({'error':type(error).__name__}),flush=True)
        time.sleep(20)


if __name__=='__main__':run()
