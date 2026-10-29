'use client';
import {useEffect,useRef,useState} from 'react';
import type Hls from 'hls.js';
import {RefreshCw,Volume2,VolumeX} from 'lucide-react';

export default function LivePlayer({camera,onState,minimal=false}:{camera:string;onState?:(state:string)=>void;minimal?:boolean}){
  const ref=useRef<HTMLVideoElement>(null);
  const [status,setStatus]=useState('Connecting'),[attempt,setAttempt]=useState(0),[muted,setMuted]=useState(true);
  useEffect(()=>{
    const listen=(event:Event)=>{const active=(event as CustomEvent<string>).detail===camera;if(minimal){setMuted(!active);if(active)void ref.current?.play().catch(()=>{});}else if(!active)setMuted(true);};
    window.addEventListener('aviary-listen',listen);
    return()=>window.removeEventListener('aviary-listen',listen);
  },[camera,minimal]);
  useEffect(()=>{
    const video=ref.current;if(!video||!camera)return;
    let hls:Hls|undefined,retry:number|undefined,lastTime=0,unchanged=0,destroyed=false,restartPending=false;
    const source=`/api/cameras/${camera}/hls/live.m3u8`;
    const report=(event:string,detail='')=>{void fetch(`/api/cameras/${camera}/playback`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({event,detail})}).catch(()=>{});};
    const play=()=>{void video.play().catch(()=>{if(!destroyed)setStatus('Tap play to watch');});};
    const restart=()=>{if(!destroyed&&!restartPending){restartPending=true;report('restart',`time=${video.currentTime.toFixed(2)} ready=${video.readyState}`);setAttempt(x=>x+1);}};
    const playing=()=>{setStatus('');onState?.('playing');report('playing');};
    const waiting=()=>onState?.('buffering');
    const ended=()=>{setStatus('Reconnecting');restart();};
    const error=()=>{setStatus('Reconnecting');window.clearTimeout(retry);retry=window.setTimeout(restart,3000);};
    video.addEventListener('playing',playing);video.addEventListener('waiting',waiting);video.addEventListener('error',error);video.addEventListener('ended',ended);
    async function connect(){
      const native=!!video!.canPlayType('application/vnd.apple.mpegurl');
      const webkit=/AppleWebKit/.test(navigator.userAgent)&&!/(Chrome|Chromium|Edg|OPR|Android)\//.test(navigator.userAgent);
      // Safari uses its native player. Chromium can now advertise native HLS,
      // but our same-stream comparison found repeated native decoder resets.
      // Prefer the measured MSE path there; retain native as a fallback.
      if(native&&webkit){video!.dataset.playbackEngine='native-hls';report('engine','native-hls');video!.src=source;play();return;}
      const {default:Engine}=await import('hls.js');if(destroyed)return;
      if(!Engine.isSupported()){if(native){video!.dataset.playbackEngine='native-hls';video!.src=source;play();}else setStatus('Try Safari or Chrome to watch live.');return;}
      video!.dataset.playbackEngine='hls.js';report('engine','hls.js');
      hls=new Engine({lowLatencyMode:false,liveSyncDurationCount:3,liveMaxLatencyDurationCount:8,maxLiveSyncPlaybackRate:1.05,maxBufferLength:20,backBufferLength:10,manifestLoadingMaxRetry:6,levelLoadingMaxRetry:6,fragLoadingMaxRetry:6});
      hls.loadSource(source);hls.attachMedia(video!);hls.on(Engine.Events.MANIFEST_PARSED,play);
      hls.on(Engine.Events.ERROR,(_,data)=>{
        report('error',`${data.type} ${data.details} fatal=${data.fatal} ${data.error?.message||''}`.slice(0,600));
        if(data.fatal){if(data.type===Engine.ErrorTypes.MEDIA_ERROR)hls?.recoverMediaError();else error();}
      });
    }
    void connect().catch(()=>{if(!destroyed)error();});
    const timer=window.setInterval(()=>{
      // The wall has no pause control. A stalled source can leave a live video
      // paused/ended; it must still be eligible for recovery when foregrounded.
      // Preserve a deliberate pause in the player that exposes native controls.
      if(document.hidden||(!minimal&&video.paused))return;
      if(minimal&&video.paused&&unchanged===2)play();
      unchanged=video.currentTime===lastTime?unchanged+1:0;lastTime=video.currentTime;
      if(unchanged===4){setStatus('Buffering');onState?.('buffering');report('stall',`ready=${video.readyState} buffer=${video.buffered.length?video.buffered.end(video.buffered.length-1)-video.currentTime:0}`);}
      if(unchanged>=15)restart();
    },1000);
    return()=>{destroyed=true;window.clearTimeout(retry);window.clearInterval(timer);hls?.destroy();video.removeEventListener('playing',playing);video.removeEventListener('waiting',waiting);video.removeEventListener('error',error);video.removeEventListener('ended',ended);video.pause();video.removeAttribute('src');video.load();};
  },[camera,attempt,onState,minimal]);
  function toggleSound(){if(muted)window.dispatchEvent(new CustomEvent('aviary-listen',{detail:camera}));setMuted(!muted);}
  return <div className={'stable-player'+(minimal?' minimal-player':'')}><video ref={ref} controls={!minimal} autoPlay muted={muted} playsInline preload="auto"/>{!minimal&&<button className="listen-button" aria-label={muted?'Listen to camera':'Mute camera'} onClick={toggleSound}>{muted?<VolumeX size={16}/>:<Volume2 size={16}/>}<span>{muted?'Listen':'Sound on'}</span></button>}{status&&<div className="player-status">{!minimal&&<span>{status}</span>}<button aria-label={status+' — reconnect video'} title={status} onClick={()=>setAttempt(x=>x+1)}><RefreshCw size={15}/></button></div>}</div>;
}
