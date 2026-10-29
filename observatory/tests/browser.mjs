// Run on a test host with Playwright installed; secrets and output stay external.
// AVIARY_URL, AVIARY_PIN_FILE, AVIARY_SCREENSHOTS; optional PLAYWRIGHT_MODULE.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const origin = process.env.AVIARY_URL || 'http://127.0.0.1:8740';
const output = process.env.AVIARY_SCREENSHOTS;
if (!process.env.AVIARY_PIN_FILE || !output) throw Error('Private PIN file and screenshot directory are required');
await fs.mkdir(output, {recursive:true, mode:0o700});
const browser = await chromium.launch({channel:'chrome', headless:true});
const context = await browser.newContext({viewport:{width:390,height:844}, hasTouch:true, isMobile:true, deviceScaleFactor:1});
const page = await context.newPage();
const errors = [], failures = [], checks = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => {if (message.type() === 'error') errors.push(message.text().slice(0,400));});
async function check(name, run) {try {await run(); checks.push(name)} catch (error) {failures.push({name,error:error.message.slice(0,1500)});}}
async function shot(name) {await page.screenshot({path:path.join(output,name+'.png'),fullPage:false});}
async function noOverflow() {assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth+1), 'Horizontal page overflow');}
async function noPageScroll() {assert.ok(await page.evaluate(()=>document.documentElement.scrollHeight<=innerHeight+1&&document.body.scrollHeight<=innerHeight+1),'Page must fit the viewport');}
async function verticalSwipe(x,y,toY){
 const cdp=await context.newCDPSession(page);
 await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x,y}]});
 for(let n=1;n<=12;n++){await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x,y:y+(toY-y)*n/12}]});await page.waitForTimeout(15);}
 await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await cdp.detach();
}
async function swipe() {
  const cdp = await context.newCDPSession(page);
  await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:70,y:400}]});
  for (let x=90;x<=330;x+=20) await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x,y:400}]});
  await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
  await cdp.detach();
}
let playback;
try {
  await page.goto(origin,{waitUntil:'domcontentloaded'});
  await page.getByLabel('Home passcode',{exact:true}).fill((await fs.readFile(process.env.AVIARY_PIN_FILE,'utf8')).trim());
  await page.getByRole('button',{name:'Open Aviary',exact:true}).click();
  await page.locator('.live-wall').waitFor();
  await check('default live wall', async()=>{assert.equal(await page.locator('.live-wall video').count(),3);await noOverflow();});
  await check('Live has no page scroll',async()=>{await noPageScroll();await verticalSwipe(200,650,200);assert.equal(await page.evaluate(()=>scrollY),0);});
  await page.waitForTimeout(18000);
  const before = await page.locator('.live-wall video').evaluateAll(v => v.map(x => ({time:x.currentTime,ready:x.readyState,width:x.videoWidth,height:x.videoHeight,paused:x.paused})));
  await page.waitForTimeout(8000);
  const after = await page.locator('.live-wall video').evaluateAll(v => v.map(x => ({time:x.currentTime,ready:x.readyState,width:x.videoWidth,height:x.videoHeight,paused:x.paused,quality:x.getVideoPlaybackQuality().toJSON?.() || {dropped:x.getVideoPlaybackQuality().droppedVideoFrames,total:x.getVideoPlaybackQuality().totalVideoFrames}})));
  playback = {before,after};
  await check('all live feeds advance',async()=>{assert.ok(after.every((v,i)=>v.ready>=2&&v.width>0&&v.time>before[i].time+2),JSON.stringify(playback));});
  await shot('phone-live');
  await check('one audible camera',async()=>{
    await page.getByRole('button',{name:'Listen to selected camera'}).click();
    assert.equal(await page.locator('.live-wall video').evaluateAll(v=>v.filter(x=>!x.muted).length),1);
    await page.getByLabel('Camera for sound and movement').selectOption({index:1});
    assert.equal(await page.locator('.live-wall video').evaluateAll(v=>v.filter(x=>!x.muted).length),1);
    await page.getByRole('button',{name:'Mute live audio'}).click();
  });
  await check('right swipe opens Reels',async()=>{await swipe();await page.locator('.reels-feed').waitFor({timeout:4000});await noOverflow();});
  if (!await page.locator('.reels-feed').count()) await page.locator('.mobile-nav').getByRole('button',{name:'Reels',exact:true}).click();
  await page.getByRole('button',{name:'Recent video',exact:true}).click();
  await page.locator('.reel video').first().waitFor();
  await page.waitForTimeout(5000);
  await shot('phone-reels');
  await check('Reels video plays',async()=>{assert.ok(await page.locator('.reel video').first().evaluate(v=>v.currentTime>0&&v.videoWidth>0&&!v.paused));await noOverflow();});
  await check('Reels snaps exactly one clip without page scroll',async()=>{
   await noPageScroll();const height=await page.locator('.reels-stack').evaluate(e=>e.clientHeight);
   await verticalSwipe(200,690,250);await page.waitForTimeout(1200);
   const scroll=await page.locator('.reels-stack').evaluate(e=>e.scrollTop);
   assert.ok(Math.abs(scroll-height)<=3,`Expected one ${height}px clip; scroll=${scroll}`);
   assert.equal(await page.evaluate(()=>scrollY),0);
  });
  await check('right swipe opens Feed',async()=>{await swipe();await page.locator('.activity-feed').waitFor({timeout:4000});await noOverflow();});
  if (!await page.locator('.activity-feed').count()) await page.locator('.mobile-nav').getByRole('button',{name:'Feed',exact:true}).click();
  await page.locator('.activity-feed article').first().waitFor({timeout:10000});
  await check('Feed images load',async()=>{await page.waitForFunction(()=>Array.from(document.querySelectorAll('.social-photo img')).some(x=>x.complete&&x.naturalWidth>0),{},{timeout:15000});});
  await shot('phone-feed');
  await page.getByRole('button',{name:'Daily five tagging',exact:true}).click();
  await page.locator('.learning-frame img').waitFor();
  await page.waitForTimeout(1000);
  await shot('phone-learning');
  await check('learning full frame loads without overflow',async()=>{assert.ok(await page.locator('.learning-frame img').evaluate(x=>x.complete&&x.naturalWidth>0));await noOverflow();});
  await page.locator('.mobile-nav').getByRole('button',{name:'Live',exact:true}).click();
  await page.setViewportSize({width:844,height:390});
  await page.waitForTimeout(5000);
  await shot('phone-landscape');
  await check('landscape grid fits',async()=>{await noOverflow();const box=await page.locator('.live-wall-grid').boundingBox();assert.ok(box&&box.height>100&&box.y>=0&&box.y+box.height<390);});
  await page.setViewportSize({width:1024,height:768});
  await page.waitForTimeout(3000);await shot('tablet-live');
  await check('tablet grid fits',noOverflow);
  await page.setViewportSize({width:360,height:640});
  await check('small phone Live fits without scroll',async()=>{await noPageScroll();await noOverflow();const boxes=await page.locator('.live-wall-tile').evaluateAll(items=>items.map(e=>{const r=e.getBoundingClientRect();return{top:r.top,bottom:r.bottom,height:r.height}}));assert.ok(boxes.every(r=>r.top>=58&&r.bottom<=528&&r.height>0));});
} catch (error) {failures.push({name:'browser smoke',error:error.message.slice(0,2000)});await shot('failure').catch(()=>{});}
finally {await fs.writeFile(path.join(output,'browser-results.json'),JSON.stringify({checks,failures,errors,playback},null,2));await browser.close();}
console.log(JSON.stringify({checks,failures,errors,playback},null,2));
process.exitCode = failures.length ? 1 : 0;
