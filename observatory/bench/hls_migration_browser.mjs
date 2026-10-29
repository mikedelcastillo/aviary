// Offline migration against an already-playing, unchanged hls.js instance.
// Node's own deadline still runs if the renderer main thread becomes stuck.
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import assert from 'node:assert/strict';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const [fixtureFolder,hlsPath,mode,output]=process.argv.slice(2);
const files=(await fs.readdir(fixtureFolder)).filter(n=>/^s\d+\.ts$/.test(n)).sort();
const segments=await Promise.all(files.slice(0,9).map(n=>fs.readFile(path.join(fixtureFolder,n))));
const hls=await fs.readFile(hlsPath);let phase=0,changedAt=0;
const started=Date.now(),base=mode==='high-contiguous'?1793247441657148:1793247000;
const next=mode==='huge'?1793247441657148:base+6;
const oldCC=mode==='high-contiguous'?20755:0,newCC=mode==='huge'?20755:oldCC+1;
function manifest(){
 const count=phase?Math.min(9,3+Math.floor((Date.now()-changedAt)/2000)):6;
 const first=phase?next:base,cc=phase?newCC:oldCC;
 const date=phase?changedAt-6000:started-12000;
 const lines=['#EXTM3U','#EXT-X-VERSION:6','#EXT-X-TARGETDURATION:2',`#EXT-X-MEDIA-SEQUENCE:${first}`,`#EXT-X-DISCONTINUITY-SEQUENCE:${cc}`,'#EXT-X-INDEPENDENT-SEGMENTS'];
 for(let i=0;i<count;i++)lines.push('#EXTINF:2.000000,',`#EXT-X-PROGRAM-DATE-TIME:${new Date(date+i*2000).toISOString()}`,`${phase?'new':'old'}-${i}.ts`);
 return lines.join('\n')+'\n';
}
const server=http.createServer((req,res)=>{
 res.setHeader('Cache-Control','no-store');
 if(req.url==='/hls.js'){res.setHeader('Content-Type','text/javascript');res.end(hls);return}
 if(req.url==='/live.m3u8'){res.setHeader('Content-Type','application/vnd.apple.mpegurl');res.end(manifest());return}
 const match=req.url.match(/^\/(?:old|new)-(\d+)\.ts$/);
 if(match){res.setHeader('Content-Type','video/mp2t');res.end(segments[Number(match[1])]);return}
 res.setHeader('Content-Type','text/html');res.end('<!doctype html><video muted autoplay></video><script src="/hls.js"></script>');
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const browser=await chromium.launch({channel:'chrome',headless:true});const page=await browser.newPage();
const deadline=(promise,ms)=>Promise.race([promise,new Promise((_,reject)=>setTimeout(()=>reject(Error('renderer-response-timeout')),ms))]);
let result={mode,base,next,oldCC,newCC};
try{
 await page.goto(`http://127.0.0.1:${server.address().port}`);
 await page.evaluate(()=>{
  const video=document.querySelector('video');window.errors=[];window.frags=[];window.ticks=0;
  setInterval(()=>window.ticks++,100);
  window.hls=new Hls({lowLatencyMode:false,liveSyncDurationCount:3,liveMaxLatencyDurationCount:8,maxLiveSyncPlaybackRate:1.05,maxBufferLength:20,backBufferLength:10,manifestLoadingMaxRetry:6,levelLoadingMaxRetry:6,fragLoadingMaxRetry:6});
  hls.on(Hls.Events.ERROR,(_,d)=>errors.push({detail:d.details,fatal:d.fatal}));
  hls.on(Hls.Events.FRAG_CHANGED,(_,d)=>frags.push({sn:d.frag.sn,cc:d.frag.cc}));
  hls.loadSource('/live.m3u8');hls.attachMedia(video);hls.on(Hls.Events.MANIFEST_PARSED,()=>video.play());
 });
 await page.waitForFunction(()=>document.querySelector('video').currentTime>1&&document.querySelector('video').getVideoPlaybackQuality().totalVideoFrames>10,null,{timeout:15000});
 result.before=await page.evaluate(()=>({time:document.querySelector('video').currentTime,ticks,frags:[...frags]}));
 phase=1;changedAt=Date.now();
 await new Promise(resolve=>setTimeout(resolve,12000));
 try{result.after=await deadline(page.evaluate(()=>({time:document.querySelector('video').currentTime,ticks,frags:[...frags],errors:[...errors]})),3000);result.responsive=true}
 catch(error){result.responsive=false;result.error=error.message}
 await fs.writeFile(output,JSON.stringify(result,null,2));console.log(JSON.stringify(result));
 if(mode!=='huge'){
  assert.equal(result.responsive,true);
  assert.ok(result.after.time>result.before.time+6,'Playback advances across migration');
  assert.equal(result.after.frags.at(-1)?.cc,newCC,'Player enters the replacement generation');
 }else{
  assert.ok(!result.responsive||!result.after.frags.some(f=>f.cc===newCC),'The original huge jump reproduces failure to enter the new generation');
 }
}finally{await browser.close();server.closeAllConnections();server.close()}
