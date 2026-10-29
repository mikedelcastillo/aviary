// Run on the isolated browser build host, with Playwright and Chrome installed.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {fileURLToPath} from 'node:url';
import {withDeadline} from './browser_deadline.mjs';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');

test('renderer hang times out while the independent Node heartbeat advances',async()=>{
  const browser=await chromium.launch({channel:'chrome',headless:true});
  let ticks=0,timer;
  try{
    const page=await browser.newPage();
    assert.equal(await withDeadline(page.evaluate(()=>42),2000),42);
    timer=setInterval(()=>ticks++,50);
    await assert.rejects(withDeadline(page.evaluate(()=>{while(true){}}),500,'fixture-hang'),
      {name:'ObservationDeadlineError',operation:'fixture-hang'});
    assert.ok(ticks>=3,'heartbeat must continue outside the blocked renderer');
  }finally{
    clearInterval(timer);
    await withDeadline(browser.close(),5000,'fixture-close');
  }
});

test('complete observer writes a failed observation and heartbeat for a hung fixture',async()=>{
  const scratch=await fs.mkdtemp(path.join(os.tmpdir(),'aviary-observer-deadline-'));
  const server=http.createServer((request,response)=>{
    if(request.url==='/api/version'){
      response.setHeader('Content-Type','application/json');response.end('{"version":"fixture"}');return;
    }
    response.setHeader('Content-Type','text/html');
    response.end(`<label>Home passcode<input></label><button onclick="document.querySelector('.live-wall').hidden=false;setTimeout(()=>{while(true){}},700)">Open Aviary</button><div class="live-wall" hidden>Fixture</div>`);
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  let child;
  try{
    const pin=path.join(scratch,'pin.txt'),output=path.join(scratch,'observation.jsonl');
    await fs.writeFile(pin,'synthetic-fixture',{mode:0o600});
    const script=fileURLToPath(new URL('./observe_browser.mjs',import.meta.url));
    child=spawn(process.execPath,[script,'--url',`http://127.0.0.1:${server.address().port}`,
      '--pin-file',pin,'--output',output,'--seconds','5','--interval','0.1','--evaluate-timeout','0.3'],{stdio:'ignore'});
    const [code]=await withDeadline(once(child,'exit'),10000,'observer-process');
    assert.notEqual(code,0);
    const rows=(await fs.readFile(output,'utf8')).trim().split('\n').map(JSON.parse);
    assert.ok(rows.some(row=>Array.isArray(row.videos)),'must record responsive samples before hang');
    assert.equal(rows.at(-1).fatal_error,'ObservationDeadlineError');
    assert.equal(rows.at(-1).operation,'playback-evaluate');
    assert.equal(rows.at(-1).complete,false);
    const heartbeat=JSON.parse(await fs.readFile(output+'.heartbeat.json','utf8'));
    assert.equal(heartbeat.phase,'failed');
    assert.equal(heartbeat.fatal_error,'ObservationDeadlineError');
    assert.ok(heartbeat.samples>0);
  }finally{
    if(child?.exitCode===null)child.kill('SIGKILL');
    server.closeAllConnections();
    await new Promise(resolve=>server.close(resolve));
    await fs.rm(scratch,{recursive:true,force:true});
  }
});
