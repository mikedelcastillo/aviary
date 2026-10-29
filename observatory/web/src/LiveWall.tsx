'use client';
import {useEffect,useState} from 'react';
import {Crosshair,MapPin,Volume2,VolumeX,X} from 'lucide-react';
import type {Camera} from './types';
import LivePlayer from './LivePlayer';
import PTZControls from './PTZControls';

export default function LiveWall({cameras,zones}:{cameras:Camera[];zones:()=>void}){
 const[selected,setSelected]=useState(cameras[0]?.id||''),[sound,setSound]=useState(false),[controls,setControls]=useState(false);
 const camera=cameras.find(c=>c.id===selected)||cameras[0];
 useEffect(()=>{if(!camera||camera.ptz_supported===false)setControls(false)},[camera?.id,camera?.ptz_supported]);
 function listen(id:string){window.dispatchEvent(new CustomEvent('aviary-listen',{detail:id}));}
 function choose(id:string){setSelected(id);if(cameras.find(c=>c.id===id)?.ptz_supported===false)setControls(false);if(sound)listen(id)}
 return <div className="live-wall">
  <div className={'live-wall-grid '+(cameras.length>3?'many':'few')} data-count={cameras.length} aria-label="All cameras live">
   {cameras.map(c=><div key={c.id} className={'live-wall-tile '+(c.id===camera?.id&&controls?'selected':'')} onClick={()=>choose(c.id)}><LivePlayer camera={c.id} minimal/></div>)}
  </div>
  {controls&&camera&&camera.ptz_supported!==false&&<div className="wall-ptz"><button className="wall-ptz-close" aria-label="Close camera controls" onClick={()=>setControls(false)}><X size={18}/></button><PTZControls key={camera.id} camera={camera.id}/></div>}
  <div className="wall-toolbar">
   <select aria-label="Camera for sound and movement" value={camera?.id||''} onChange={e=>choose(e.target.value)}>{cameras.map(c=><option key={c.id} value={c.id}>{c.name}</option>)}</select>
   <button aria-label={sound?'Mute live audio':'Listen to selected camera'} aria-pressed={sound} onClick={()=>{listen(sound?'':camera?.id||'');setSound(!sound)}}>{sound?<Volume2 size={21}/>:<VolumeX size={21}/>}</button>
   <button aria-label={camera?.ptz_supported===false?'This camera has no movement motor':'Move selected camera'} disabled={camera?.ptz_supported===false} aria-pressed={controls} onClick={()=>setControls(!controls)}><Crosshair size={21}/></button>
   <button aria-label="Feeder and water zones" onClick={zones}><MapPin size={21}/></button>
  </div>
 </div>
}
