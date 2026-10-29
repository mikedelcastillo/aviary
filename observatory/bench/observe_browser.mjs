// Development-host playback observation. Output and PIN file must stay private.
// Playwright/Chrome are supplied by an isolated benchmark environment.
import fs from 'node:fs/promises';
import path from 'node:path';
import {parseArgs} from 'node:util';
import {setTimeout as delay} from 'node:timers/promises';
import {withDeadline} from './browser_deadline.mjs';

const {values} = parseArgs({options:{
  url:{type:'string',default:'http://127.0.0.1:8740'},
  'pin-file':{type:'string'}, output:{type:'string'},
  seconds:{type:'string',default:'7200'}, interval:{type:'string',default:'15'},
  build:{type:'string',default:'unspecified'},
  'force-hlsjs':{type:'boolean',default:false},
  'evaluate-timeout':{type:'string',default:'10'},
}});
if(!values['pin-file'] || !values.output) throw Error('--pin-file and --output are required');
const seconds=Number(values.seconds), interval=Number(values.interval);
const evaluateTimeout=Number(values['evaluate-timeout']);
if(!Number.isFinite(seconds)||seconds<0||!Number.isFinite(interval)||interval<=0||!Number.isFinite(evaluateTimeout)||evaluateTimeout<=0) throw Error('Invalid durations');
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
await fs.mkdir(path.dirname(values.output),{recursive:true,mode:0o700});
const output=await fs.open(values.output,'wx',0o600);
const heartbeatPath=values.output+'.heartbeat.json';
let heartbeat={phase:'launching',observer_pid:process.pid,samples:0,last_sample_utc:null};
let heartbeatWrite=null;
async function writeHeartbeat(){
  if(heartbeatWrite)return heartbeatWrite;
  heartbeatWrite=(async()=>{
    const temporary=heartbeatPath+'.tmp';
    await fs.writeFile(temporary,JSON.stringify({...heartbeat,timestamp:new Date().toISOString()}),{mode:0o600});
    await fs.rename(temporary,heartbeatPath);
  })();
  try{await heartbeatWrite;}finally{heartbeatWrite=null;}
}
await writeHeartbeat();
// This timer runs in Node, outside the renderer being measured.
const heartbeatTimer=setInterval(()=>{void writeHeartbeat().catch(()=>{});},2000);
heartbeatTimer.unref();
const browser=await chromium.launch({channel:'chrome',headless:true});
const context=await browser.newContext({viewport:{width:390,height:844},hasTouch:true,isMobile:true,deviceScaleFactor:1,timezoneId:'Asia/Manila'});
const page=await context.newPage();
let pageErrors=0, consoleErrors=0, failedRequests=0, shuttingDown=false;
page.on('pageerror',()=>pageErrors++);
page.on('console',message=>{if(message.type()==='error')consoleErrors++;});
page.on('requestfailed',()=>failedRequests++);
process.on('SIGTERM',()=>{shuttingDown=true;});
process.on('SIGINT',()=>{shuttingDown=true;});

// An explicit test-only canary can compare the application's existing MSE path
// against native Chrome HLS without changing the deployed application.
if(values['force-hlsjs'])await page.addInitScript(()=>{
  const original=HTMLMediaElement.prototype.canPlayType;
  HTMLMediaElement.prototype.canPlayType=function(type){
    if(['application/vnd.apple.mpegurl','application/x-mpegurl'].includes(type.toLowerCase()))return '';
    return original.call(this,type);
  };
});

// This observes HTMLMediaElement state only. It never calls play(), changes
// media sources, presses reconnect controls, reloads, or moves a camera.
await page.addInitScript(()=>{
  const records=new Map();
  let serial=0;
  const events=['play','playing','waiting','stalled','pause','ended','seeking','seeked','emptied','loadedmetadata','error','canplay'];
  const attach=()=>{
    for(const video of document.querySelectorAll('video')){
      if(records.has(video))continue;
      const record={id:++serial,created:Date.now(),counts:{},recent:[]};
      records.set(video,record);
      for(const event of events)video.addEventListener(event,()=>{
        record.counts[event]=(record.counts[event]||0)+1;
        record.recent.push({event,time:Date.now(),currentTime:video.currentTime,ready:video.readyState,paused:video.paused});
        if(record.recent.length>30)record.recent.shift();
      });
    }
  };
  window.__aviaryPlaybackSample=()=>{
    attach();
    const videos=[...document.querySelectorAll('.live-wall video')];
    return {hidden:document.hidden,viewport:{width:innerWidth,height:innerHeight},
      overflow:document.documentElement.scrollWidth>innerWidth+1,
      videos:videos.map((video,slot)=>{
        const record=records.get(video),quality=video.getVideoPlaybackQuality();
        const buffer=[];
        for(let i=0;i<video.buffered.length;i++)buffer.push([video.buffered.start(i),video.buffered.end(i)]);
        return {slot:slot+1,elementId:record.id,created:record.created,
          time:video.currentTime,ready:video.readyState,network:video.networkState,
          width:video.videoWidth,height:video.videoHeight,paused:video.paused,ended:video.ended,
          muted:video.muted,volume:video.volume,rate:video.playbackRate,
          sourceKind:video.currentSrc.startsWith('blob:')?'mse-blob':video.currentSrc?'url':'none',
          error:video.error?.code||null,buffer,
          frames:quality.totalVideoFrames,dropped:quality.droppedVideoFrames,corrupted:quality.corruptedVideoFrames,
          eventCounts:{...record.counts},recentEvents:[...record.recent]};
      })};
  };
  addEventListener('DOMContentLoaded',()=>{
    attach();
    new MutationObserver(attach).observe(document.documentElement,{childList:true,subtree:true});
  });
});

let samples=0;
try{
  heartbeat.phase='login';
  await page.goto(values.url,{waitUntil:'domcontentloaded'});
  await page.getByLabel('Home passcode',{exact:true}).fill((await fs.readFile(values['pin-file'],'utf8')).trim());
  await page.getByRole('button',{name:'Open Aviary',exact:true}).click();
  await page.locator('.live-wall').waitFor();
  const observedVersion=await withDeadline(page.evaluate(async()=>{
    try{const response=await fetch('/api/version',{cache:'no-store'});return (await response.json()).version||null;}
    catch{return null;}
  }),evaluateTimeout*1000,'version-evaluate');
  const assets=await withDeadline(page.evaluate(()=>[...document.scripts].map(script=>{try{return new URL(script.src).pathname;}catch{return '';}}).filter(Boolean)),evaluateTimeout*1000,'assets-evaluate');
  const started=performance.now();
  let previous=null;
  for(let index=0;;index++){
    heartbeat.phase='sampling';
    heartbeat.evaluation_started_utc=new Date().toISOString();
    const snapshot=await withDeadline(page.evaluate(()=>window.__aviaryPlaybackSample()),evaluateTimeout*1000,'playback-evaluate');
    const elapsed=(performance.now()-started)/1000;
    for(const video of snapshot.videos){
      const old=previous?.videos.find(value=>value.slot===video.slot);
      video.observedReset=Boolean(old&&(old.elementId!==video.elementId||video.time<old.time-0.5||video.frames<old.frames));
      video.frameDelta=old&&old.elementId===video.elementId?video.frames-old.frames:null;
      video.timeDelta=old&&old.elementId===video.elementId?video.time-old.time:null;
      video.advanced=old?video.frameDelta>0&&video.timeDelta>0.25:null;
    }
    const complete=elapsed>=seconds;
    await output.write(JSON.stringify({schema:1,timestamp:new Date().toISOString(),index,
      elapsed_seconds:Number(elapsed.toFixed(3)),planned_seconds:seconds,expected_build:values.build,
      forced_hlsjs:values['force-hlsjs'],
      observed_version_at_start:observedVersion,
      assets:index===0?assets:undefined,pageErrors,consoleErrors,failedRequests,
      ...snapshot,complete,interrupted:shuttingDown})+'\n');
    samples++;
    heartbeat={...heartbeat,phase:complete?'complete':'waiting',samples,
      last_sample_utc:new Date().toISOString(),last_elapsed_seconds:Number(elapsed.toFixed(3))};
    if(complete||shuttingDown)break;
    previous=snapshot;
    await delay(Math.max(0,Math.min((index+1)*interval,seconds)-elapsed)*1000);
  }
}catch(error){
  heartbeat={...heartbeat,phase:'failed',fatal_error:error.name,operation:error.operation||null};
  await output.write(JSON.stringify({timestamp:new Date().toISOString(),fatal_error:error.name,
    operation:error.operation||null,complete:false,samples,pageErrors,consoleErrors,failedRequests})+'\n');
  throw error;
}finally{
  clearInterval(heartbeatTimer);
  if(heartbeatWrite)await heartbeatWrite;
  await writeHeartbeat();
  await output.close();
  await withDeadline(browser.close(),5000,'browser-close');
}
console.log(JSON.stringify({samples,finished:new Date().toISOString(),interrupted:shuttingDown}));
