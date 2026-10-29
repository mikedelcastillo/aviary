import React,{useEffect,useRef,useState} from 'react';
import {ArrowLeft,ArrowRight,ArrowUp,ArrowDown,House,Square,MapPin} from 'lucide-react';
import type {LucideIcon} from 'lucide-react';
import {errorMessage,type Zone} from './types';
const actions:[string,LucideIcon][]=[['left',ArrowLeft],['up',ArrowUp],['down',ArrowDown],['right',ArrowRight],['home',House],['stop',Square]];
export default function PTZControls({camera}:{camera:string}){
 const[busy,setBusy]=useState(false),[stopping,setStopping]=useState(false),[error,setError]=useState(''),[zones,setZones]=useState<Zone[]>([]);
 const moving=useRef(false),stopPending=useRef(false),generation=useRef(0),latest=useRef(0);
 useEffect(()=>{
  const version=++generation.current,abort=new AbortController();
  moving.current=false;stopPending.current=false;setBusy(false);setStopping(false);setError('');setZones([]);
  fetch('/api/zones',{signal:abort.signal}).then(r=>r.json()).then(z=>{
   if(generation.current===version)setZones(Array.isArray(z)?z.filter(x=>x.camera===camera&&x.anchor):[]);
  }).catch(()=>{});
  return()=>{generation.current++;abort.abort()};
 },[camera]);
 async function move(action:string,zone?:string){
  const stop=action==='stop';
  if(stop?stopPending.current:moving.current||stopPending.current)return;
  const version=generation.current,request=++latest.current;
  if(stop){stopPending.current=true;setStopping(true)}else{moving.current=true;setBusy(true)}
  setError('');
  try{
   const r=await fetch(zone?`/api/zones/${zone}/look`:`/api/cameras/${camera}/ptz`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});
   if(!r.ok){const detail=await r.json().catch(()=>null);throw Error(typeof detail?.detail==='string'?detail.detail:'The camera did not accept the command.')}
  }catch(e){if(generation.current===version&&latest.current===request)setError(errorMessage(e))}
  finally{
   if(generation.current===version){
    if(stop){stopPending.current=false;setStopping(false)}else{moving.current=false;setBusy(false)}
   }
  }
 }
 return <div className="ptz-panel"><span className="ptz-label">CAMERA</span><div className="ptz-arrows">{actions.map(([action,Icon])=><button key={action} aria-label={`Camera ${action}`} title={action} disabled={action==='stop'?stopping:busy||stopping} onClick={()=>void move(action)}><Icon size={18}/></button>)}</div>{zones.map(z=><button key={z.id} className="zone-shortcut" disabled={busy||stopping} onClick={()=>void move('anchor',z.id)}><MapPin size={14}/>{z.name}</button>)}{error&&<span className="ptz-error" role="alert">{error}</span>}</div>
}
