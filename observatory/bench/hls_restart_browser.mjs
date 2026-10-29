// Real, unmodified hls.js. No player replacement, page reload or custom recovery.
import fs from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const [url,output]=process.argv.slice(2);
const browser=await chromium.launch({channel:'chrome',headless:true});
const page=await browser.newPage();
try{
 await page.goto(url);
 await page.evaluate(()=>{
  window.result={version:Hls.version,players:[],samples:[],events:[]};
  for(let i=1;i<=3;i++){
   const video=document.createElement('video');video.autoplay=true;video.muted=true;video.playsInline=true;document.body.append(video);
   const hls=new Hls({lowLatencyMode:false,liveSyncDurationCount:3,liveMaxLatencyDurationCount:8,maxLiveSyncPlaybackRate:1.05,maxBufferLength:20,backBufferLength:10,manifestLoadingMaxRetry:6,levelLoadingMaxRetry:6,fragLoadingMaxRetry:6});
   const state={id:`fixture-${i}`,hls,video,errors:[],fragments:[],ended:0};window.result.players.push(state);
   hls.on(Hls.Events.ERROR,(_,data)=>state.errors.push({at:performance.now(),type:data.type,detail:data.details,fatal:data.fatal}));
   hls.on(Hls.Events.FRAG_CHANGED,(_,data)=>state.fragments.push({at:performance.now(),sn:data.frag.sn,cc:data.frag.cc,pdt:data.frag.programDateTime}));
   video.addEventListener('ended',()=>state.ended++);
   hls.loadSource(`/hls/fixture-${i}/live.m3u8`);hls.attachMedia(video);
   hls.on(Hls.Events.MANIFEST_PARSED,()=>video.play().catch(()=>{}));
  }
  window.sampling=setInterval(()=>{
   const at=performance.now();
   for(const {id,hls,video} of window.result.players)window.result.samples.push({at,id,time:video.currentTime,paused:video.paused,ready:video.readyState,frames:video.getVideoPlaybackQuality().totalVideoFrames,latency:hls.latency,edge:hls.mainForwardBufferInfo?.end??null});
  },250);
 });
 await page.waitForFunction(()=>window.result.players.every(p=>p.video.currentTime>1&&p.video.getVideoPlaybackQuality().totalVideoFrames>5),{timeout:20000});
 await fs.writeFile(output+'/browser-ready','ready');
 const deadline=Date.now()+130000;
 while(Date.now()<deadline){
  try{await fs.access(output+'/finish');break}catch{}
  await new Promise(resolve=>setTimeout(resolve,1000));
 }
 const result=await page.evaluate(()=>{
  clearInterval(window.sampling);
  return {version:window.result.version,events:window.result.events,samples:window.result.samples,players:window.result.players.map(({id,errors,fragments,ended,video,hls})=>({id,errors,fragments,ended,time:video.currentTime,frames:video.getVideoPlaybackQuality().totalVideoFrames,paused:video.paused,ready:video.readyState,latency:hls.latency}))};
 });
 await fs.writeFile(output+'/browser-result.json',JSON.stringify(result,null,2));
 console.log(JSON.stringify({hlsVersion:result.version,players:result.players.map(p=>({id:p.id,time:p.time,frames:p.frames,latency:p.latency,fatal:p.errors.filter(e=>e.fatal).length,ended:p.ended,finalCC:p.fragments.at(-1)?.cc}))}));
}finally{await browser.close()}
