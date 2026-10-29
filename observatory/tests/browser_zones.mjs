// Isolated static bundle + fictional API routes. Never creates a real zone or
// contacts a camera. Run with AVIARY_URL pointed at a local static test server.
import assert from 'node:assert/strict';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const browser=await chromium.launch({channel:'chrome',headless:true});
const context=await browser.newContext({viewport:{width:390,height:844},hasTouch:true,isMobile:true,serviceWorkers:'block'});
const page=await context.newPage(), touch=await context.newCDPSession(page);
const cameras=['example','other'].map(id=>({id,name:id==='example'?'Fictional front':'Fictional rear',online:true,age:1,privacy:'clear',ptz_supported:false}));
const saves=[],requests=[],errors=[];
let serial=0,fail=false,shortExpiry=false,delayed=null,zones=[];
const image=n=>`<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="${n%2?'#254c39':'#455672'}"/><circle cx="320" cy="240" r="70" fill="#dfd1ac"/><text x="20" y="35" fill="white">Fictional frozen view ${n}</text></svg>`;
await page.route('**/api/**',async route=>{
 const request=route.request(),path=new URL(request.url()).pathname;
 requests.push({path,method:request.method()});
 if(path==='/api/session')return route.fulfill({json:{authenticated:true}});
 if(path==='/api/overview')return route.fulfill({json:{cameras,birds:[],timezone:'UTC',archive:{count:0,first:null,last:null}}});
 if(path==='/api/version')return route.fulfill({json:{version:'fixture'}});
 if(path==='/api/events')return route.fulfill({json:{events:[],next:null}});
 if(path.endsWith('/zone-reference')){
  const number=++serial,camera=path.split('/')[3];
  if(delayed){const wait=delayed;delayed=null;await wait}
  if(fail)return route.fulfill({status:503,json:{detail:'Fictional privacy pause. Please refresh.'}});
  return route.fulfill({contentType:'image/svg+xml',headers:{'X-Aviary-Zone-Reference':`${camera}-reference-${number}`,'X-Aviary-Zone-Ttl':shortExpiry?'2':'300'},body:image(number)});
 }
 if(path==='/api/zones'){
  if(request.method()==='POST'){
   const data=request.postDataJSON();saves.push(data);zones.push({...data,id:'zone-'+saves.length});
   return route.fulfill({json:{id:'zone-'+saves.length}});
  }
  return route.fulfill({json:zones});
 }
 if(path.endsWith('/playback'))return route.fulfill({json:{ok:true}});
 if(path.includes('/hls/'))return route.fulfill({status:503,body:'Fictional video unavailable'});
 throw Error(`Unexpected fixture API request: ${request.method()} ${path}`);
});
page.on('pageerror',error=>errors.push(error.message));
const save=page.getByRole('button',{name:'Save Zone',exact:true});
const refresh=page.getByRole('button',{name:'Refresh picture',exact:true});
async function ready(){await page.getByText('Frozen picture ready. Draw and save within five minutes.',{exact:true}).waitFor()}
async function newPicture(previous){
 await page.waitForFunction(old=>{const picture=document.querySelector('img[alt="Frozen screened picture for this zone"]');return picture&&picture.src!==old&&picture.naturalWidth>0},previous);
 await ready();
}
async function draw(){
 const canvas=page.getByLabel('Zone drawing canvas',{exact:true});
 await canvas.evaluate(e=>e.scrollIntoView({block:'center',behavior:'instant'}));
 const box=await canvas.boundingBox();
 assert.ok(Math.abs(box.width/box.height-4/3)<.02,'Canvas matches the picture aspect ratio');
 const point=t=>({x:box.x+box.width*(.2+.4*t),y:box.y+box.height*(.2+.5*t),id:1,radiusX:2,radiusY:2,force:1});
 assert.ok(await page.evaluate(p=>!!document.elementFromPoint(p.x,p.y)?.closest('.zone-canvas'),point(0)),'Touch starts inside the canvas');
 await touch.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[point(0)]});
 for(let i=1;i<=6;i++){await touch.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[point(i/6)]});await page.waitForTimeout(20)}
 await touch.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
 await page.getByRole('textbox',{name:'Zone name',exact:true}).fill('Fictional bowl');
 assert.equal(await save.isEnabled(),true);
}
try{
 await page.goto((process.env.AVIARY_URL||'http://127.0.0.1:8766')+'/?tab=live');
 await page.getByRole('button',{name:'Feeder and water zones',exact:true}).click();
 await ready();await draw();
 const shown=await page.getByAltText('Frozen screened picture for this zone',{exact:true}).getAttribute('src');
 assert.ok(shown.startsWith('blob:'));
 await save.click();
 await newPicture(shown);
 assert.equal(saves.length,1);assert.equal(saves[0].reference_token,'example-reference-1');
 assert.ok(Math.abs(saves[0].rect[0]-.2)<.02&&Math.abs(saves[0].rect[2]-.4)<.02);
 assert.notEqual(await page.getByAltText('Frozen screened picture for this zone',{exact:true}).getAttribute('src'),shown);
 await draw();
 let release;delayed=new Promise(resolve=>{release=resolve});
 await refresh.click();
 await page.getByText('Loading a stable screened picture…',{exact:true}).waitFor();
 assert.equal(await save.isDisabled(),true,'Refresh invalidates the old rectangle and token immediately');
 release();await ready();assert.equal(await save.isDisabled(),true);
 await draw();
 const beforeSwitch=await page.getByAltText('Frozen screened picture for this zone',{exact:true}).getAttribute('src');
 await page.getByRole('combobox',{name:'Zone camera',exact:true}).selectOption('other');
 await newPicture(beforeSwitch);assert.equal(await save.isDisabled(),true,'Camera switch clears the rectangle');
 await draw();
 const beforeSave=await page.getByAltText('Frozen screened picture for this zone',{exact:true}).getAttribute('src');
 await save.click();await newPicture(beforeSave);
 assert.equal(saves[1].camera,'other');assert.ok(saves[1].reference_token.startsWith('other-reference-'));
 fail=true;await refresh.click();
 await page.getByText('Fictional privacy pause. Please refresh.',{exact:true}).waitFor();
 assert.equal(await save.isDisabled(),true,'Screening failure cannot save an older view');
 assert.equal(await page.getByAltText('Frozen screened picture for this zone',{exact:true}).count(),0);
 fail=false;shortExpiry=true;await refresh.click();await ready();await draw();
 await page.getByText('This picture has expired. Refresh and draw the zone again.',{exact:true}).waitFor();
 assert.equal(await save.isDisabled(),true,'Expired reference clears the rectangle and token');
 assert.equal(saves.length,2);
 assert.equal(requests.some(r=>r.path.endsWith('/snapshot')||r.path.endsWith('/ptz')||r.path.endsWith('/position')),false);
 assert.deepEqual(errors,[]);
 assert.ok(await page.locator('html').evaluate(e=>e.scrollWidth<=innerWidth+1));
 console.log(JSON.stringify({fixtureOnly:true,passed:['mobile touch drawing','exact token sent on save','actual image aspect ratio','fresh reference after save','refresh invalidates old selection','camera-bound reference','privacy pause fails closed','expiry blocks save','no snapshot or movement calls']}));
}catch(error){await page.screenshot({path:'zone-fixture-failure.png',fullPage:true});throw error}
finally{await browser.close()}
