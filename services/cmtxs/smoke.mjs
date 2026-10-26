// End-to-end protocol check against Wrangler local or a deployed Worker.
import assert from 'node:assert/strict';
import {WebSocket} from 'ws';
import {recommendSwap,recommendQue} from '../../source/cmtxs/engine.js';
const base=process.argv[2]||'http://127.0.0.1:8787',origin='https://tan1-z.github.io';
async function post(url,name){const r=await fetch(base+url,{method:'POST',headers:{'Content-Type':'application/json',Origin:origin},body:JSON.stringify({name})});assert.equal(r.status,200,await r.clone().text());assert.equal(r.headers.get('Access-Control-Allow-Origin'),origin);return r.json();}
function client(c){
  const ws=new WebSocket(base.replace(/^http/,'ws')+`/rooms/${c.code}/socket`,{origin}),waiters=[];let latest=null;
  const wait=(predicate,timeout=15000)=>new Promise((resolve,reject)=>{if(latest&&predicate(latest))return resolve(latest);const w={predicate,resolve,reject};w.timer=setTimeout(()=>{waiters.splice(waiters.indexOf(w),1);reject(Error('State timeout'));},timeout);waiters.push(w);});
  const c1={ws,wait,get state(){return latest;},async command(action,extra={},id=crypto.randomUUID()){
    const ack=wait(m=>(m.type==='ack'||m.type==='error')&&m.id===id);ws.send(JSON.stringify({type:'command',id,decision:latest?.decision,action,...extra}));const m=await ack;if(m.type==='error')throw Error(m.message);return m;
  }};
  ws.on('open',()=>ws.send(JSON.stringify({type:'auth',token:c.token})));ws.on('error',()=>{});
  ws.on('message',data=>{if(data.toString()==='pong')return;const m=JSON.parse(data);if(m.type==='state')latest=m;for(const w of [...waiters])if(w.predicate(m)){clearTimeout(w.timer);waiters.splice(waiters.indexOf(w),1);w.resolve(m);}});return c1;
}
const bad=await fetch(base+'/health',{headers:{Origin:'https://unrelated.invalid'}});assert.equal(bad.status,403);
const a=await post('/rooms','联机验证甲'),b=await post(`/rooms/${a.code}/join`,'联机验证乙');
let clients=[client(a),client(b)],recovered;
try{
  await Promise.all(clients.map(c=>c.wait(m=>m.type==='state'&&m.members.every(p=>p.online))));
  for(const c of clients)await c.command('ready');await clients[0].command('start');
  await Promise.all(clients.map(c=>c.wait(m=>m.type==='state'&&m.phase==='swap')));
  const own=[...clients[0].state.game.players[0].hand];clients[0].ws.close();await clients[1].wait(m=>m.type==='state'&&!m.members[0].online);
  recovered=client(a);clients.push(recovered);await recovered.wait(m=>m.type==='state');assert.deepEqual(recovered.state.game.players[0].hand,own);
  const id=crypto.randomUUID(),indices=recommendSwap(own);await recovered.command('swap',{indices},id);assert.equal((await recovered.command('swap',{indices},id)).duplicate,true);
  await clients[1].command('swap',{indices:recommendSwap(clients[1].state.game.players[1].hand)});
  await Promise.all([recovered,clients[1]].map(c=>c.wait(m=>m.type==='state'&&m.phase==='que')));
  await recovered.command('que',{suit:recommendQue(recovered.state.game.players[0].hand)});await clients[1].command('que',{suit:recommendQue(clients[1].state.game.players[1].hand)});
  await recovered.wait(m=>m.type==='state'&&m.phase==='discard');
  assert.equal(recovered.state.game.players[1].hand,null);assert.equal(clients[1].state.game.players[0].hand,null);
  await recovered.command('auto',{value:true});await clients[1].command('auto',{value:true});console.log('PASS: CORS, 2 seats, reconnect, duplicate suppression, private views, swap and dingque');
  const timer=setInterval(()=>{const s=recovered.state;console.log('Playing:',s.phase,'wall',s.game.remaining);},25000);
  let end;try{end=await recovered.wait(m=>m.type==='state'&&m.phase==='ended',180000);}finally{clearInterval(timer);}
  const other=await clients[1].wait(m=>m.type==='state'&&m.phase==='ended');assert.deepEqual(end.game.ledger,other.game.ledger);assert.equal(end.game.players.reduce((sum,p)=>sum+p.score,0),0);assert.ok(end.review.every(r=>r.seat===0));assert.ok(other.review.every(r=>r.seat===1));
  console.log('PASS: complete game, identical ledgers, zero-sum settlement, seat-specific review');
}finally{for(const c of clients)c.ws.terminate();}
