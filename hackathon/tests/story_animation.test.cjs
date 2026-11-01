const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../ui/story.js'), 'utf8');
const chartData = () => ({kind:'bsl-chart',width:744,height:1215,image:'/assets/bsl-alphabet.png',glyphs:Object.fromEntries([...new Set('QUIETBRIDGE')].map(letter=>[letter,{crop:[12,9,103,100]}]))});
function harness({reduced=false,fetch=async()=>({ok:true,json:async()=>chartData()})}={}) {
  const classes=new Set(),handlers={},timers=[],cards=[];
  const classList={add:(...names)=>names.forEach(name=>classes.add(name)),remove:(...names)=>names.forEach(name=>classes.delete(name))};
  const next={hidden:false};
  const hands={hidden:false,append:tile=>cards.push(tile)};
  const element=()=>({append(){},getContext:()=>({drawImage(){}}),animate:()=>({cancel(){}})});
  let fetchCount=0;
  vm.runInNewContext(source,{
    document:{readyState:'complete',documentElement:{classList},body:{classList},getElementById:id=>id==='story-next'?next:hands,addEventListener:(event,fn)=>{handlers[event]=fn;},createElement:element},
    window:{matchMedia:()=>({matches:reduced,addEventListener(){}})},
    Image:class{naturalWidth=744;naturalHeight=1215;set src(value){queueMicrotask(()=>this.onload?.());}},
    fetch:async(...args)=>{fetchCount++;return fetch(...args);},
    setTimeout:(handler,delay)=>{const timer={handler,delay,cancelled:false};timers.push(timer);return timer;},clearTimeout:timer=>{if(timer)timer.cancelled=true;}
  });
  return {classes,next,hands,cards,handlers,timers,get fetchCount(){return fetchCount;}};
}
const flush=()=>new Promise(resolve=>setImmediate(resolve));
test('reduced motion immediately reveals word and Next without fetching',async()=>{
  const h=harness({reduced:true});await flush();assert.equal(h.fetchCount,0);assert.equal(h.next.hidden,false);assert.equal(h.classes.has('is-complete'),true);assert.equal(h.timers.length,0);
});
test('eleven hand letters hold four seconds and Next appears only after the flip',async()=>{
  const h=harness();assert.equal(h.next.hidden,true);await flush();assert.equal(h.cards.length,11);assert.equal(h.classes.has('is-illustrating'),true);
  h.timers.find(t=>t.delay===4000).handler();assert.equal(h.classes.has('is-settling'),true);assert.equal(h.next.hidden,true);
  h.timers.find(t=>t.delay===900).handler();assert.equal(h.classes.has('is-complete'),true);assert.equal(h.next.hidden,false);assert.equal(h.hands.hidden,true);
});
test('Escape before data arrives reveals Next and prevents delayed animation',async()=>{
  let resolve;const pending=new Promise(done=>{resolve=done;});const h=harness({fetch:()=>pending});h.handlers.keydown({key:'Escape'});resolve({ok:true,json:async()=>chartData()});await flush();assert.equal(h.next.hidden,false);assert.equal(h.cards.length,0);assert.equal(h.classes.has('is-illustrating'),false);
});
test('failed illustration data reveals readable word and Next',async()=>{
  const h=harness({fetch:async()=>{throw new Error('offline');}});await flush();assert.equal(h.classes.has('is-complete'),true);assert.equal(h.next.hidden,false);
});
test('unavailable chart response reveals readable word and Next',async()=>{
  const h=harness({fetch:async()=>({ok:false})});await flush();assert.equal(h.next.hidden,false);assert.equal(h.classes.has('is-complete'),true);
});
test('a stalled asset request cannot permanently hide Next',async()=>{
  const h=harness({fetch:()=>new Promise(()=>{})});h.timers.find(t=>t.delay===10000).handler();assert.equal(h.next.hidden,false);assert.equal(h.classes.has('is-complete'),true);
});
