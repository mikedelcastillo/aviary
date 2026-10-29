import {errorMessage,type Stats} from './types';
import type {LucideIcon} from 'lucide-react';
import React,{useEffect,useState} from 'react';
import {Activity,Database,HardDrive,Cpu,Radio} from 'lucide-react';

const readinessReasons:Record<string,string>={
 playlist_missing:'Playlist missing',playlist_stale:'Playlist stale',playlist_invalid:'Playlist invalid',playlist_unavailable:'Playlist unavailable',
 capture_missing:'Capture missing',capture_stale:'Capture stale',capture_invalid:'Capture invalid',capture_unavailable:'Capture unavailable',
 recorder_missing:'Recorder heartbeat missing',recorder_stale:'Recorder heartbeat stale',recorder_invalid:'Recorder heartbeat invalid',recorder_unavailable:'Recorder heartbeat unavailable',
 camera_invalid:'Camera configuration invalid',
};
const age=(value:number|null|undefined)=>typeof value==='number'&&Number.isFinite(value)?value.toFixed(1)+' s':'Unknown';

export default function NerdStats(){
 const[data,setData]=useState<Stats|null>(null),[error,setError]=useState('');
 useEffect(()=>{let active=true;async function update(){try{const r=await fetch('/api/stats');if(!r.ok)throw Error('Stats unavailable');const value=await r.json();if(active){setData(value);setError('')}}catch(e){if(active)setError(errorMessage(e))}}void update();const timer=setInterval(()=>{if(!document.hidden)void update()},5000);return()=>{active=false;clearInterval(timer)}},[]);
 if(error)return <p role="alert">{error}</p>;
 if(!data)return <p className="muted">Reading service counters…</p>;
 const memory=data.services.reduce((s,v)=>s+(v.memory_mib||0),0),cores=data.services.reduce((s,v)=>s+(v.cpu_cores||0),0),max=Math.max(...data.coverage.map(x=>x.count),1);
 return <div className="nerd-stats">
  <div className="stats-summary">{([[Cpu,'Service CPU',cores.toFixed(2)+' cores'],[Activity,'Service memory',memory.toFixed(0)+' MiB'],[Database,'Indexed sightings',data.events.events.toLocaleString()],[HardDrive,'Disk available',data.disk_free_gib+' GiB']] as [LucideIcon,string,string][]).map(([Icon,label,value])=><div key={label}><Icon size={19}/><span>{label}</span><strong>{value}</strong></div>)}</div>
  <p className="stats-note">Measured every 5 seconds. CPU is logical cores in use. Service memory includes child processes; the shared Ollama model process is additional.</p>
  <section><h2>Service processes</h2><p className="stats-note">Active means the service process is running. Fresh video output is checked separately below.</p><div className="stats-table-wrap"><table><thead><tr><th>Service</th><th>Process state</th><th>RAM / MiB</th><th>CPU / cores</th><th>Restarts</th></tr></thead><tbody>{data.services.map(s=><tr key={s.name}><td>{s.name}</td><td><span className={'dot '+(s.state==='active'?'on':'')}/> {s.state}</td><td>{s.memory_mib??'—'}</td><td>{s.cpu_cores??'Measuring'}</td><td>{s.restarts}</td></tr>)}</tbody></table></div></section>
  <section><h2>Camera pipeline</h2><p className="stats-note">Video readiness needs a playlist younger than 12 seconds, a capture younger than 8 seconds, and a recorder heartbeat younger than 6 seconds. Vision freshness is a separate signal; these checks do not verify playback on your device.</p><div className="pipeline-grid">{data.cameras.map(c=><article key={c.id}>
   <h3><Radio size={17}/>{c.name}</h3><dl>
    <dt>Video output</dt><dd>{c.video_ready===true?'Fresh':c.video_ready===false?'Not ready':'Not measured'}</dd>
    {!!c.video_reasons?.length&&<><dt>Readiness</dt><dd>{c.video_reasons.map(reason=><div key={reason}>{readinessReasons[reason]||'Freshness unavailable'}</div>)}</dd></>}
    <dt>Playlist age</dt><dd>{age(c.playlist_age)}</dd><dt>Capture age</dt><dd>{age(c.capture_age)}</dd><dt>Recorder heartbeat</dt><dd>{age(c.recorder_age)}</dd>
    <dt>Vision output</dt><dd>{c.online?'Fresh':'Unavailable or stale'}</dd><dt>Vision frame age</dt><dd>{age(c.age)}</dd>
    <dt>Vision latency</dt><dd>{c.inference_ms??'Withheld'} ms</dd><dt>Reconnects</dt><dd>{c.reconnects??'—'}</dd><dt>Current tracks</dt><dd>{c.detections?.length||0}</dd><dt>Privacy screen</dt><dd>{c.privacy}</dd><dt>Rolling segments</dt><dd>{c.segments}</dd><dt>Recording storage</dt><dd>{c.recording_mib} MiB</dd>
   </dl>
  </article>)}</div></section>
  <section><h2>Recorded sightings / last 24 hours</h2><div className="coverage-chart" aria-label="Hourly recorded sightings">{data.coverage.map(x=><div key={x.hour} title={`${x.hour}: ${x.count} sightings`}><i style={{height:Math.max(2,x.count/max*110)+'px'}}/><span>{x.hour.slice(-2)}</span></div>)}</div><p className="stats-note">UTC hours. Counts are sampled records, not continuous observation coverage.</p></section>
  <section className="judging-stats"><div><h2>Video review decisions</h2><dl>{data.decisions.map(x=><React.Fragment key={x.status}><dt>{x.status}</dt><dd>{x.count}</dd></React.Fragment>)}</dl></div><div><h2>Local processing</h2><dl><dt>Archive importer</dt><dd>{data.archive_engine}</dd><dt>Archive index</dt><dd>{data.db_mib} MiB</dd><dt>VLM</dt><dd>{data.curator_model}</dd><dt>Automatic highlights</dt><dd>{data.auto_publish?'Enabled':'Under evaluation'}</dd><dt>Host load / 1m</dt><dd>{data.host_load[0]}</dd><dt>Logical CPU threads</dt><dd>{data.cpu_threads}</dd></dl></div></section>
  {data.last_judge&&<section><h2>Latest visual judgment</h2><p>{data.last_judge.caption}</p><p className="stats-note">{data.last_judge.status} · {data.last_judge.analysis.model} · {data.last_judge.analysis.latency_seconds}s</p><details><summary>Evidence & uncertainty</summary><pre>{JSON.stringify(data.last_judge.analysis,null,2)}</pre></details></section>}
 </div>;
}
