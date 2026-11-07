#!/usr/bin/env node
// Isolated, contract-level browser checks. No camera, audio, downloads or shared browser.
import {spawn} from 'node:child_process';
import {mkdtemp, rm, realpath, access} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';

const root = path.resolve(import.meta.dirname, '..');
const sleep = ms => new Promise(r => setTimeout(r, ms));
let deadline;
let currentCase = 'startup', origin = '', snapshotId = 'unavailable';
function fail(expected, actual, checkId) { throw Object.assign(new Error(`case=${currentCase} expected=${JSON.stringify(expected)} actual=${JSON.stringify(actual)} origin=${origin} snapshotID=${snapshotId}`), {caseId:currentCase,checkId}); }
function expect(ok, expected, actual, checkId) { if (!ok) fail(expected, actual, checkId); }
async function waitFor(fn, expected, ms=5000) {
  const end = Math.min(deadline, Date.now()+ms); let actual;
  while (Date.now() < end) { actual = await fn(); if (actual) return actual; await sleep(100); }
  fail(expected, actual);
}
function lineProtocol(child) {
  let buffer='', id=0; const pending=new Map(), ready=[];
  child.stdout.setEncoding('utf8');
  child.stdout.on('data', chunk => { buffer+=chunk; let n; while ((n=buffer.indexOf('\n'))>=0) { const line=buffer.slice(0,n); buffer=buffer.slice(n+1); if(!line.trim()) continue; let msg; try { msg=JSON.parse(line); } catch { continue; } if(msg.ready) ready.push(msg); else if(pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); } } });
  child.stderr.on('data', b => process.stderr.write(`[fixture] ${b}`));
  return {ready:()=>waitFor(()=>ready.shift(),'fixture ready',10000), async command(op) { const key=++id; const p=new Promise(resolve=>pending.set(key,resolve)); child.stdin.write(JSON.stringify({id:key,op})+'\n'); const result=await Promise.race([p,sleep(5000).then(()=>null)]); expect(result?.ok, `${op} fixture response`,result); return result; }};
}
function cdpProtocol(child) {
  let buffer=Buffer.alloc(0), id=0; const pending=new Map(), listeners=new Map();
  child.stdio[4].on('data', chunk=>{buffer=Buffer.concat([buffer,chunk]); let n; while((n=buffer.indexOf(0))>=0) {const data=buffer.subarray(0,n).toString(); buffer=buffer.subarray(n+1); let m; try {m=JSON.parse(data);} catch {continue;} if(m.id && pending.has(m.id)){const [resolve,reject]=pending.get(m.id);pending.delete(m.id);m.error?reject(new Error(JSON.stringify(m.error))):resolve(m.result);} else if(m.method) for(const f of listeners.get(m.method)||[]) f(m.params,m.sessionId);}});
  child.stderr.on('data',()=>{});
  return {send(method,params={},sessionId) {const key=++id; const p=new Promise((resolve,reject)=>pending.set(key,[resolve,reject])); child.stdio[3].write(JSON.stringify({id:key,method,params,...(sessionId?{sessionId}:{})})+'\0'); return Promise.race([p,sleep(10000).then(()=>{throw Error(`CDP timeout ${method}`);})]);}, on(method,fn){if(!listeners.has(method)) listeners.set(method,[]);listeners.get(method).push(fn);}};
}
async function chromePath() {
  if(process.env.CHROME_BIN) return process.env.CHROME_BIN;
  const choices=['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome','/usr/bin/google-chrome','/usr/bin/chromium','/usr/bin/chromium-browser'];
  for(const p of choices) try {await access(p);return p;} catch {}
  throw Error('Chrome not found; set CHROME_BIN');
}
async function run(variant, coreOnly=false) {
  deadline=Date.now()+85000;
  let fixture, browser, profile, protocol, cdp, session;
  try {
    currentCase='fixture-start';
    fixture=spawn(process.env.PYTHON_BIN || path.join(root,'.venv/bin/python'),['-u','scripts/web_check_fixture.py','--variant',variant],{cwd:root,stdio:['pipe','pipe','pipe']});
    fixture.on('error', e=>process.stderr.write(`fixture spawn: ${e.message}\n`));
    protocol=lineProtocol(fixture); const ready=await protocol.ready(); origin=ready.url;
    expect(/^http:\/\/127\.0\.0\.1:\d+\/$/.test(origin),'owned loopback URL',origin);
    expect(ready.variant===variant,'requested variant',ready.variant);
    profile=await mkdtemp(path.join(tmpdir(),'quietbridge-web-check-'));
    browser=spawn(await chromePath(),['--headless=new','--remote-debugging-pipe',`--user-data-dir=${profile}`,'--no-first-run','--no-default-browser-check','about:blank'],{stdio:['ignore','ignore','pipe','pipe','pipe']});
    browser.on('error', e=>process.stderr.write(`Chrome spawn: ${e.message}\n`));
    cdp=cdpProtocol(browser);
    const target=await cdp.send('Target.createTarget',{url:'about:blank'});
    session=(await cdp.send('Target.attachToTarget',{targetId:target.targetId,flatten:true})).sessionId;
    const send=(method,params={})=>cdp.send(method,params,session);
    await send('Page.enable'); await send('Runtime.enable'); await send('Debugger.enable');
    const parsedScripts=[];
    cdp.on('Debugger.scriptParsed',(p,s)=>{if(s===session && p.url===new URL('/app.js',origin).href) parsedScripts.push(p.scriptId);});
    const dialogs=[]; cdp.on('Page.javascriptDialogOpening',(p,s)=>{if(s===session) dialogs.push(p);});
    const evaluate=async expression=>{const result=await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true}); if(result.exceptionDetails) throw Error(result.exceptionDetails.text+': '+JSON.stringify(result.exceptionDetails.exception)); return result.result.value;};
    const js=async fn=>evaluate(`(${fn})()`);
    const click=async selector=>{await evaluate(`setTimeout(()=>document.querySelector(${JSON.stringify(selector)}).click(),0)`);};
    const edit=async value=>{await evaluate(`(()=>{let e=document.querySelector('#corrected-text');e.value=${JSON.stringify(value)};e.dispatchEvent(new Event('input',{bubbles:true}));})()`);};
    const state=()=>evaluate("fetch('/api/state').then(r=>r.json())");
    const snap=()=>protocol.command('snapshot');
    const visible=()=>js(()=>({message:document.querySelector('#message')?.textContent,editor:document.querySelector('#corrected-text')?.value,speakDisabled:document.querySelector('#speak')?.disabled,saveDisabled:document.querySelector('#save-correction')?.disabled,error:document.querySelector('#error')?.textContent,alternatives:[...document.querySelectorAll('.alternative')].map(x=>({text:x.dataset.candidate,label:x.textContent,selected:x.getAttribute('aria-pressed')}))}));
    const dialog=async accept=>{await waitFor(()=>dialogs.length,'native confirmation dialog');const d=dialogs.shift();expect(d.type==='confirm','native confirm',d);await send('Page.handleJavaScriptDialog',{accept});};
    const action=async (name,data)=>evaluate(`(async()=>fetch('/api/action/${name}',{method:'POST',headers:{'Content-Type':'application/json','X-Lipflow-Token':(await fetch('/api/state').then(r=>r.json())).token},body:JSON.stringify(${JSON.stringify(data)})}).then(async r=>({ok:r.ok,body:await r.json()})))()`);
    await send('Page.navigate',{url:origin});
    await waitFor(async()=>{try{return (await state()).token && await js(()=>!!document.querySelector('#corrected-text'));}catch{return false;}},'page and state ready',10000);
    currentCase='identity';
    const app=await (await fetch(new URL('/app.js',origin))).arrayBuffer();
    const hash=createHash('sha256').update(Buffer.from(app)).digest('hex');
    snapshotId=`${ready.identity.commit}:${hash}`;
    expect(hash===ready.identity.files['app.js'],'served /app.js SHA-256 matches fixture identity',{hash,identity:ready.identity.files['app.js']});
    await waitFor(()=>parsedScripts.length,'browser parsed the expected app.js');
    const loaded=await send('Debugger.getScriptSource',{scriptId:parsedScripts.at(-1)});
    const executedHash=createHash('sha256').update(loaded.scriptSource).digest('hex');
    expect(executedHash===hash,'browser-loaded app.js equals served snapshot',{executedHash,hash});
    const browserVersion=await cdp.send('Browser.getVersion');
    console.log(JSON.stringify({variant,origin,snapshotID:snapshotId,identity:ready.identity,browser:browserVersion.product,evidence:'real DOM/HTTP; silent speech dispatch; some DOM-dispatched input; no physical capture or audible speech'}));
    currentCase='typed-speech';
    await waitFor(async()=> (await visible()).editor !== undefined && (await state()).token && await js(()=>typeof state !== 'undefined' && !!state),'client initial render');
    await edit('Please bring water'); await waitFor(async()=>!(await visible()).saveDisabled,'typed apply enabled'); await click('#save-correction');
    await waitFor(async()=> (await state()).decision.text==='Please bring water' && !(await visible()).speakDisabled,'typed message applied');
    let before=await snap(); expect(before.speech_calls.length===0,'no automatic speech',before.speech_calls);
    await click('#speak'); await waitFor(async()=> (await snap()).speech_calls.length===1,'speech dispatched only after click');
    let after=await snap(); expect(after.speech_calls[0]==='Please bring water','silent speech dispatch text',after.speech_calls);
    currentCase='conflict-selection';
    let r=await action('rehearse',{text:'I am comfortable',label:'THUMB_DOWN'});
    expect(r.ok,'rehearse conflict accepted',r);
    await waitFor(async()=>{let v=await visible(); return v.alternatives.length>=2 && v.speakDisabled;},'conflict requires selection');
    let v=await visible(), s=await state();
    expect(s.decision.reason==='conflict','HTTP conflict',s.decision);
    expect(v.alternatives.every(x=>x.label===x.text && s.alternatives.includes(x.text)),'visible candidate labels match submitted candidates and HTTP', {dom:v.alternatives,http:s.alternatives},'visible-label');
    const intended='No.';
    expect(v.alternatives.filter(x=>x.label===intended).length===1,'one visible No. candidate',v);
    await evaluate(`setTimeout(()=>[...document.querySelectorAll('.alternative')].find(b=>b.textContent===${JSON.stringify(intended)}).click(),0)`);
    await waitFor(async()=>!(await visible()).speakDisabled,'selection enables confirmation');
    v=await visible(); const selected=v.alternatives.filter(x=>x.selected==='true');
    expect(selected.length===1 && selected[0].label===intended && v.editor===intended && v.message===intended,'chosen No. matches visible label, editor and message',v,'selected-editor');
    const countBeforeConfirm=(await snap()).speech_calls.length;
    await click('#speak');
    await waitFor(async()=> (await snap()).speech_calls.length===countBeforeConfirm+1,'explicit conflict confirmation dispatches');
    const confirmed=await snap();
    expect(confirmed.speech_calls.at(-1)===intended && confirmed.state.decision.text===intended,'confirmed dispatch matches the intended No.',confirmed);
    r=await action('rehearse',{text:'I am comfortable',label:'THUMB_DOWN'});expect(r.ok,'new conflict for draft tests',r);
    await waitFor(async()=> (await visible()).alternatives.length>=2 && (await visible()).speakDisabled,'fresh conflict awaiting deliberate choice');
    currentCase='draft-continuity';
    await edit('Unfinished private draft'); await click('.alternative'); await dialog(false);
    v=await visible(); expect(v.editor==='Unfinished private draft' && v.speakDisabled,'declining replacement preserves draft',v);
    await click('.alternative'); await dialog(true);
    v=await visible(); expect(v.editor!=='Unfinished private draft' && !v.speakDisabled,'accepting replacement selects candidate',v);
    await edit('Draft survives incoming result');
    r=await action('rehearse',{text:'Another incoming message',label:'THUMB_DOWN'}); expect(r.ok,'incoming rehearsal',r);
    await waitFor(async()=> (await state()).decision.reason==='conflict','incoming conflict state');
    await sleep(300); v=await visible(); expect(v.editor==='Draft survives incoming result' && v.speakDisabled,'incoming result preserves unsent draft',v,'incoming-draft');
    await click('a[href="/settings"]'); await waitFor(()=>evaluate("location.pathname==='/settings'"),'Settings navigation');
    await send('Page.navigate',{url:origin}); await waitFor(async()=>{try{return (await visible()).editor==='Draft survives incoming result';}catch{return false;}},'draft restored after Settings navigation');
    currentCase='pending-confirm-edit';
    await edit('Draft before pending confirm');
    await click('.alternative:last-child'); await dialog(true);
    await waitFor(async()=>!(await visible()).speakDisabled,'selected candidate ready to confirm');
    const callsBefore=(await snap()).speech_calls.length;
    await evaluate(`(()=>{const original=window.fetch.bind(window);window.__originalFetch=original;window.__confirmRelease=null;window.fetch=(u,o)=>String(u).endsWith('/api/action/confirm') ? new Promise((resolve,reject)=>{window.__confirmRelease=()=>original(u,o).then(resolve,reject);}) : original(u,o);})()`);
    await click('#speak'); await waitFor(()=>evaluate('!!window.__confirmRelease'),'confirmation request held');
    await edit('New words during confirmation'); await evaluate('window.__confirmRelease()');
    await waitFor(async()=> (await state()).decision.status==='accepted','confirmation reached server');
    await sleep(500);
    const callsAfter=(await snap()).speech_calls;
    expect(callsAfter.length===callsBefore && (await visible()).editor==='New words during confirmation','no old speech after editing during confirmation',{before:callsBefore,after:callsAfter,dom:await visible()},'edited-confirm-dispatch');
    await evaluate('window.fetch=window.__originalFetch');
    console.log(`PASS core journeys variant=${variant}`);
    if(coreOnly) return;
    currentCase='pending-new-message';
    // Delay one actual fetch before it reaches HTTP; release sends the original request.
    await evaluate(`(()=>{const original=window.fetch.bind(window);window.__originalFetch=original;window.__release=null;window.fetch=(u,o)=>{if(String(u).endsWith('/api/action/new_message')) return new Promise((resolve,reject)=>{window.__release=()=>original(u,o).then(resolve,reject);o.signal?.addEventListener('abort',()=>reject(new DOMException('Aborted','AbortError')),{once:true});});return original(u,o);};})()`);
    await click('#new-message'); await dialog(true); await waitFor(()=>evaluate('!!window.__release'),'new message request delayed');
    await edit('Newer edit while request pending'); await evaluate('window.__release()');
    await waitFor(async()=>!(await visible()).saveDisabled,'pending new message completed');
    v=await visible(); expect(v.editor==='Newer edit while request pending','newer edit survives delayed new-message HTTP',v);
    await evaluate('window.fetch=window.__originalFetch');
    currentCase='restart-draft';
    const old=await state(); await protocol.command('restart');
    await waitFor(async()=>{try {let x=await state();return x.server_instance!==old.server_instance && x.token!==old.token;}catch{return false;}},'new server instance and token',8000);
    await waitFor(async()=>{let x=await visible();return x.editor==='Newer edit while request pending' && !x.saveDisabled && x.error.includes('restarted');},'old tab retains applicable draft after restart',8000);
    await click('#save-correction'); await waitFor(async()=> (await state()).decision.text==='Newer edit while request pending','draft applied with fresh token');
    currentCase='timeout-no-retry';
    await edit('Retry only manually');
    await evaluate(`(()=>{const original=window.fetch.bind(window);window.__originalFetch=original;window.__blocked=0;window.fetch=(u,o)=>{if(String(u).endsWith('/api/action/correct')){window.__blocked++;return new Promise((resolve,reject)=>o.signal?.addEventListener('abort',()=>reject(new DOMException('Aborted','AbortError')),{once:true}));}return original(u,o);};})()`);
    const start=Date.now(); await click('#save-correction');
    r=await action('rehearse',{text:'Polling remains live',label:'THUMB_DOWN'}); expect(r.ok,'incoming HTTP during blocked correction',r);
    await waitFor(async()=> (await visible()).alternatives.length>=2,'poll renders incoming result while POST blocked');
    await waitFor(async()=> (await visible()).error.includes('timed out'),'real action timeout',17500);
    const elapsed=Date.now()-start, attempts=await evaluate('window.__blocked');
    expect(elapsed>=14500 && attempts===1,'15 second timeout without retry',{elapsed,attempts});
    v=await visible(); expect(v.editor==='Retry only manually','timeout keeps draft',v);
    await evaluate('window.fetch=window.__originalFetch'); await click('#save-correction');
    await waitFor(async()=> (await state()).decision.text==='Retry only manually','manual retry reaches HTTP');
    currentCase='viewport-keyboard';
    for(const [width,height] of [[390,844],[640,900]]) {
      await send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
      const metrics=await send('Page.getLayoutMetrics'); const dims=await js(()=>({innerWidth,scrollWidth:document.documentElement.scrollWidth,clientWidth:document.documentElement.clientWidth}));
      expect(dims.innerWidth===width && dims.scrollWidth<=dims.clientWidth,`viewport ${width} without horizontal overflow`,{dims,metrics:metrics.cssLayoutViewport});
    }
    await evaluate("if (!document.querySelector('#first-use-guide').hidden) document.querySelector('#guide-dismiss').click(); document.querySelector('#open-guide').focus()");
    await send('Input.dispatchKeyEvent',{type:'keyDown',key:' ',code:'Space',windowsVirtualKeyCode:32}); await send('Input.dispatchKeyEvent',{type:'keyUp',key:' ',code:'Space',windowsVirtualKeyCode:32});
    expect(await js(()=>!document.querySelector('#first-use-guide').hidden),'Space opens guide from keyboard',await js(()=>document.activeElement?.id));
    await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27}); await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
    expect(await js(()=>document.querySelector('#first-use-guide').hidden && document.activeElement?.id==='open-guide'),'Escape closes guide and restores focus',await js(()=>document.activeElement?.id));
    currentCase='lab-modal';
    await send('Page.navigate',{url:new URL('/record',origin).href});
    await waitFor(async()=>{try{return await js(()=>!!document.querySelector('#btnNewProfile'));}catch{return false;}},'owned lab loaded',10000);
    await click('#btnNewProfile');
    expect(await js(()=>!document.querySelector('#newProfileDialog').hidden && document.activeElement?.id==='newProfileName'),'lab dialog opens with focus',await js(()=>document.activeElement?.id));
    await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Tab',code:'Tab',windowsVirtualKeyCode:9}); await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Tab',code:'Tab',windowsVirtualKeyCode:9});
    expect(await js(()=>document.querySelector('#newProfileDialog').contains(document.activeElement)),'Tab remains inside lab modal',await js(()=>document.activeElement?.id));
    await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27}); await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
    expect(await js(()=>document.querySelector('#newProfileDialog').hidden && document.activeElement?.id==='btnNewProfile'),'Escape closes modal and restores focus',await js(()=>document.activeElement?.id));
    console.log('PASS focused browser contracts (dispatched input; real keyboard and native dialogs; silent fixture)');
  } finally {
    if(cdp && browser && browser.exitCode===null) {try {await Promise.race([cdp.send('Browser.close'),sleep(1000)]);} catch {} }
    const waitExit=async child=>{if(child.exitCode!==null || child.signalCode!==null) return; await Promise.race([new Promise(resolve=>child.once('exit',resolve)),sleep(2000)]); if(child.exitCode===null && child.signalCode===null){child.kill('SIGKILL'); await new Promise(resolve=>child.once('exit',resolve));}};
    if(browser) {if(browser.exitCode===null && browser.signalCode===null) browser.kill(); await waitExit(browser);}
    if(protocol && fixture && fixture.exitCode===null && fixture.signalCode===null) {try {await protocol.command('stop');} catch {} }
    if(fixture) await waitExit(fixture);
    if(profile) {const parent=await realpath(tmpdir());const actual=await realpath(profile).catch(()=>null);if(actual && path.dirname(actual)===parent && path.basename(actual).startsWith('quietbridge-web-check-')) await rm(actual,{recursive:true,force:true}); else if(actual) throw Error(`Refusing to remove non-owned profile ${actual}`);}
  }
}
async function main() {
  if(process.argv.includes('--calibrate')) {
    await run('control',true);
    for(const [variant,expected,check] of [['conflict-stale','conflict-selection','selected-editor'],['draft-loss','draft-continuity','incoming-draft'],['confirm-edit-stale','pending-confirm-edit','edited-confirm-dispatch'],['choice-label-mismatch','conflict-selection','visible-label']]) {
      let found; try {await run(variant,true);} catch(e) {found={caseId:e.caseId,checkId:e.checkId}; console.log(`calibration ${variant}: ${e.message}`);}
      currentCase='calibration'; expect(found?.caseId===expected && found?.checkId===check,`${variant} fails ${expected}/${check}, not a timeout or unrelated assertion`,found);
    }
    console.log('PASS calibration sensitivity');
  } else await run('control');
}
main().catch(e=>{console.error(e.caseId?e.message:`case=${currentCase} expected=successful execution actual=${e.stack || e} origin=${origin} snapshotID=${snapshotId}`);process.exitCode=1;});
