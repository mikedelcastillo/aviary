// Real browser interaction against the deployed bundle, with isolated fake
// learning API responses. No annotations are written to the deployment.
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const browser=await chromium.launch({channel:'chrome',headless:true});
const context=await browser.newContext({viewport:{width:390,height:844},hasTouch:true,isMobile:true});
const page=await context.newPage();
const touch=await context.newCDPSession(page);
async function canvasBounds(){
 const canvas=page.getByLabel('Bird annotation canvas',{exact:true});
 await canvas.evaluate(e=>e.scrollIntoView({block:'center',behavior:'instant'}));
 const bounds=await canvas.boundingBox();
 assert.ok(await page.evaluate(({x,y})=>!!document.elementFromPoint(x,y)?.closest('[aria-label="Bird annotation canvas"]'),{x:bounds.x+bounds.width*.4,y:bounds.y+bounds.height*.15}),'Drag starts on the canvas, not a fixed header');
 return bounds;
}
async function touchDrag(box){
 const point=t=>({x:box.x+box.width*(.4+.3*t),y:box.y+box.height*(.15+.65*t),id:1,radiusX:2,radiusY:2,force:1});
 await touch.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[point(0)]});
 for(let i=1;i<=6;i++){await touch.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[point(i/6)]});await page.waitForTimeout(20)}
 await touch.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
}
const answers=[];let undone=0;
const image='<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="640" height="360" fill="#283630"/><ellipse cx="320" cy="170" rx="60" ry="85" fill="#ccc"/><text x="20" y="30" fill="#fff">Fictional review fixture</text></svg>';
const cards=Array.from({length:5},(_,i)=>({id:String(i+1).padStart(24,'0'),camera:'fixture',time:'2026-01-01T00:00:00Z',image:'/api/learn/fixture/image',video:null,frame_seconds:0,lighting:'day',reason:'audit',suggestions:[],category:'tapo/day',imported:true}));
await page.route('**/api/learn**',async route=>{
 const request=route.request(),url=new URL(request.url());
 if(url.pathname.endsWith('/image'))return route.fulfill({contentType:'image/svg+xml',body:image});
 if(url.pathname.endsWith('/undo')){undone++;return route.fulfill({json:{ok:true}})}
 if(url.pathname.endsWith('/answer')){answers.push(request.postDataJSON());return route.fulfill({json:{answer_id:'fixture-answer-'+answers.length}})}
 if(request.method()!=='GET')throw Error('Unexpected fixture mutation');
 const labels=[{name:'fern',kind:'individual'},{name:'unknown_bird',kind:'unknown'}];
 return route.fulfill({json:{cards,today:0,goal:5,pending:5,confirmed_frames:0,labels:{day:labels,ir:[],phone:labels}}});
});
try{
 await page.goto(process.env.AVIARY_URL||'http://127.0.0.1:8740');
 await page.getByLabel('Home passcode',{exact:true}).fill((await fs.readFile(process.env.AVIARY_PIN_FILE,'utf8')).trim());
 await page.getByRole('button',{name:'Open Aviary',exact:true}).click();
 await page.getByRole('button',{name:'Daily five tagging',exact:true}).click();
 await page.getByRole('button',{name:'Add a missed bird',exact:true}).waitFor();
 await page.waitForFunction(()=>document.querySelector('.learning-frame img')?.naturalWidth>0);
 await page.getByRole('button',{name:'Add a missed bird',exact:true}).click();
 let box=await canvasBounds();
 await page.touchscreen.tap(box.x+box.width*.2,box.y+box.height*.2);
 await page.touchscreen.tap(box.x+box.width*.5,box.y+box.height*.7);
 assert.equal(await page.locator('.learning-frame rect').count(),1,'Two taps create a missed-bird box');
 await page.getByRole('button',{name:'unknown bird',exact:true}).click();
 await page.getByRole('button',{name:'Redraw',exact:true}).click();
 await page.evaluate(()=>{
  window.__drawingEvents=[];
  for(const type of ['pointerdown','pointermove','pointerup','pointercancel'])document.addEventListener(type,e=>{
   if(type==='pointermove'&&!e.buttons)return;
   window.__drawingEvents.push({type,pointerType:e.pointerType,primary:e.isPrimary,x:e.clientX,y:e.clientY,target:e.target.tagName,canvas:e.target.closest('svg')?.getAttribute('aria-label'),rect:document.querySelector('[aria-label="Bird annotation canvas"]')?.getBoundingClientRect().toJSON()});
  },true);
 });
 box=await canvasBounds();
 await touchDrag(box);
 assert.equal(await page.locator('.learning-draw-hint').count(),0,'Touch drag finishes redraw');
 assert.equal(await page.locator('.learning-frame rect').count(),1,'Touch redraw replaces selected box');
 await page.getByRole('button',{name:'Redraw',exact:true}).click();
 box=await canvasBounds();
 await page.touchscreen.tap(box.x+box.width*.25,box.y+box.height*.2);
 await page.touchscreen.tap(box.x+box.width*.75,box.y+box.height*.7);
 assert.equal(await page.locator('.learning-draw-hint').count(),0,'Two-corner taps finish redraw');
 assert.equal(await page.locator('.learning-frame rect').count(),1,'Two-corner redraw replaces selected box');
 await page.getByRole('button',{name:'Redraw',exact:true}).click();
 box=await canvasBounds();
 await page.mouse.move(box.x+box.width*.4,box.y+box.height*.15);
 await page.mouse.down();await page.mouse.move(box.x+box.width*.7,box.y+box.height*.8,{steps:6});await page.mouse.up();
 assert.equal(await page.locator('.learning-frame rect').count(),1,'Redraw replaces selected box');
 await page.getByRole('checkbox').check();
 await page.getByRole('button',{name:'Save & next',exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('.learning-count')?.textContent==='1 / 5');
 assert.equal(answers[0].outcome,'annotated');assert.equal(answers[0].boxes[0].label,'unknown_bird');
 assert.ok(Math.abs(answers[0].boxes[0].cx-.55)<.02);assert.ok(Math.abs(answers[0].boxes[0].w-.3)<.02);
 await page.getByRole('button',{name:'Undo last submitted frame',exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('.learning-count')?.textContent==='0 / 5');
 assert.equal(undone,1);assert.equal(await page.locator('.learning-frame rect').count(),1);
 assert.ok(await page.getByRole('checkbox').isChecked(),'Undo restores full-frame check');
 await page.getByRole('button',{name:'Zoom 1 times. Tap to change zoom.',exact:true}).click();
 assert.ok(await page.getByRole('button',{name:'Zoom 2 times. Tap to change zoom.',exact:true}).isVisible());
 await page.getByRole('button',{name:'Remove selected box',exact:true}).click();
 assert.ok(await page.getByRole('button',{name:'Confirm empty',exact:true}).isDisabled(),'Removing a box resets full-frame confirmation');
 for(let i=0;i<5;i++){
  await page.getByRole('button',{name:'Can’t tell',exact:true}).click();
  await page.waitForFunction(n=>document.querySelector('.learning-count')?.textContent===`${n} / 5`,i+1);
 }
 await page.getByRole('heading',{name:'Your daily five are done.',exact:true}).waitFor();
 assert.equal(answers.filter(a=>a.outcome==='uncertain').length,5);
 assert.equal(await page.locator('.learning-card').count(),0,'Daily five stops at five');
 assert.ok(await page.locator('html').evaluate(e=>e.scrollWidth<=innerWidth+1));
 console.log(JSON.stringify({passed:['two-corner drawing','actual touch drag redraw','two-corner redraw','mouse drag redraw','normalized box','label preserved','undo restores edits','zoom','empty confirmation guard','five uncertain answers stop the session'],fixtureOnly:true}));
}catch(error){
 console.error(JSON.stringify({drawingEvents:await page.evaluate(()=>window.__drawingEvents||[]),checkboxDisabled:await page.getByRole('checkbox').isDisabled().catch(()=>null),drawingHint:await page.locator('.learning-draw-hint').count()}));
 await page.screenshot({path:'learning-drawing-failure.png',fullPage:true});
 throw error;
}finally{await browser.close()}
