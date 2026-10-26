import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {WebSocket} from 'ws';
import {Room} from '../room.js';
import {startServer} from '../local-server.mjs';
import {createGame,exchangeTiles,setAllQue,recommendSwap,recommendQue,assertGame,sorted} from '../../../source/cmtxs/engine.js';

function fixture(humans=2){
  let time=100000,rand=123;
  const options={now:()=>time,random:()=>{rand=(Math.imul(rand,1664525)+1013904223)>>>0;return rand/4294967296;},botDelay:5,timeout:100};
  const room=new Room(null,options),credentials=[room.init('ABCD2345','房主')];
  for(let s=1;s<humans;s++)credentials.push(room.join('牌友'+s));
  let id=0;const cmd=(s,action,extra={})=>room.command(s,{id:'test-'+(++id),decision:room.d.decision,action,...extra});
  const start=()=>{for(let s=0;s<humans;s++)cmd(s,'ready');cmd(0,'start');};
  return {room,cmd,start,credentials,options,advance:ms=>{time+=ms;return room.tick();}};
}
test('four-seat exchange and dingque validate atomically',()=>{
  const g=createGame(55),before=JSON.stringify(g),indices=g.players.map(p=>recommendSwap(p.hand));indices[3]=[0,0,1];
  assert.throws(()=>exchangeTiles(g,indices));assert.equal(JSON.stringify(g),before);
  exchangeTiles(g,g.players.map(p=>recommendSwap(p.hand)));const queued=JSON.stringify(g);
  assert.throws(()=>setAllQue(g,[0,1,2,4]));assert.equal(JSON.stringify(g),queued);
  setAllQue(g,[0,1,2,0]);assert.deepEqual(g.players.map(p=>p.que),[0,1,2,0]);assertGame(g);
});
test('lobby readiness, host start, private submissions and hidden information',()=>{
  const {room:r,cmd,start,credentials}=fixture(4);
  assert.throws(()=>cmd(1,'start'),/房主/);assert.throws(()=>cmd(0,'start'),/准备/);start();
  assert.ok(r.d.members.every(m=>!m.ready));assert.throws(()=>r.join('路人'),/已开始/);
  cmd(0,'swap',{indices:recommendSwap(r.d.game.players[0].hand)});
  const v=r.view(1),wire=JSON.stringify(v);assert.equal(v.phase,'swap');assert.equal(v.submitted,false);
  assert.ok(Array.isArray(v.game.players[1].hand));assert.equal(v.game.players[0].hand,null);
  for(const field of ['"wall"','"seed"','"events"','"submissions"','"commands"','"obs"'])assert.equal(wire.includes(field),false,field);
  for(const c of credentials)assert.equal(wire.includes(c.token),false);
  for(let s=1;s<4;s++)cmd(s,'swap',{indices:recommendSwap(r.d.game.players[s].hand)});
  assert.equal(r.d.game.phase,'que');cmd(0,'que',{suit:1});assert.ok(r.view(1).game.players.every(p=>p.que===-1));
  for(let s=1;s<4;s++)cmd(s,'que',{suit:s%3});assert.equal(r.d.game.phase,'discard');assertGame(r.d.game);
});
test('commands are exactly once, stale decisions and forged seats cannot move another hand',()=>{
  const {room:r,cmd,start}=fixture(2);start();
  const stale=r.d.decision;for(let s=0;s<2;s++)cmd(s,'swap',{indices:recommendSwap(r.d.game.players[s].hand)});
  for(let s=0;s<2;s++)cmd(s,'que',{suit:recommendQue(r.d.game.players[s].hand)});
  const before=JSON.stringify(r.serialize()),tile=r.view(0).options.discards[0];
  assert.throws(()=>r.command(0,{id:'stale',decision:stale,action:'discard',tile}),/更新/);assert.equal(JSON.stringify(r.serialize()),before);
  assert.throws(()=>cmd(1,'discard',{tile,seat:0}),/轮到/);assert.equal(JSON.stringify(r.serialize()),before);
  const c={id:'once',decision:r.d.decision,action:'discard',tile};r.command(0,c);const after=JSON.stringify(r.serialize());
  assert.equal(r.command(0,c).duplicate,true);assert.equal(JSON.stringify(r.serialize()),after);assertGame(r.d.game);
});
test('two humans can win one discard; the first reply cannot suppress the second',()=>{
  const {room:r,cmd}=fixture(4),g=createGame(1),pool=Array.from({length:108},(_,i)=>Math.floor(i/4));
  const take=t=>{const i=pool.indexOf(t);assert.notEqual(i,-1);pool.splice(i,1);return t;};
  const wait=[0,1,2,3,4,5,9,10,11,12,13,14,8];
  g.players.forEach((p,s)=>{p.hand=[];p.que=s<2?2:0;p.melds=[];p.discards=[];});
  g.players[0].hand=wait.map(take);g.players[1].hand=wait.map(take);g.players[3].discards=[take(8)];
  for(let s=2;s<4;s++)while(g.players[s].hand.length<13)g.players[s].hand.push(pool.shift());
  g.wall=pool;g.phase='respond';g.turn=3;g.last={tile:8,from:3,rob:false};g.reviewSeats=[];assertGame(g);
  r.d.game=g;r.newDecision();assert.deepEqual(r.view(0).options.responses,['hu']);
  cmd(1,'respond',{choice:'hu'});assert.equal(g.phase,'respond');assert.equal(g.winners.length,0);
  cmd(0,'respond',{choice:'hu'});assert.deepEqual(g.winners,[0,1]);assert.equal(g.players[3].score,-2);assertGame(g);
});
test('pass-hu lock survives room restart and hides the next same-fan win option',()=>{
  const {room:r,cmd,options}=fixture(4),g=createGame(1),pool=Array.from({length:108},(_,i)=>Math.floor(i/4));
  const take=t=>{const i=pool.indexOf(t);assert.notEqual(i,-1);return pool.splice(i,1)[0];};
  g.players.forEach((p,s)=>{p.hand=[];p.que=s===0?2:1;p.melds=[];p.discards=[];});
  g.players[0].hand=[0,1,2,3,4,5,9,10,11,12,13,14,15].map(take);
  g.players[1].hand=[9].map(take);g.players[2].hand=[12].map(take);g.players[3].hand=[15].map(take);const reserved=[20,21,9].map(take);
  for(let s=1;s<4;s++)while(g.players[s].hand.length<(s===1?14:13))g.players[s].hand.push(pool.shift());
  g.players.forEach(p=>p.hand=sorted(p.hand));g.wall=[...pool,...reserved.reverse()];g.phase='discard';g.turn=1;g.reviewSeats=[];g.drawn=g.players[1].hand.at(-1);r.d.game=g;r.newDecision();
  cmd(1,'discard',{tile:9});assert.ok(r.view(0).options.responses.includes('hu'));cmd(0,'respond',{choice:'pass'});
  assert.equal(g.players[0].passedHuFan,0);const restored=new Room(r.serialize(),options);
  restored.command(2,{id:'second',decision:restored.d.decision,action:'discard',tile:12});assert.ok(!restored.view(0).options.responses.includes('hu'));assert.equal(restored.view(0).game.players[0].passedHuFan,0);
  const before=JSON.stringify(restored.serialize());assert.throws(()=>restored.command(0,{id:'bad-hu',decision:restored.d.decision,action:'respond',choice:'hu'}));assert.equal(JSON.stringify(restored.serialize()),before);assertGame(restored.d.game);
});
test('timeout takeover, cancellation and restart preserve a complete mixed game',()=>{
  for(const humans of [1,2,3,4]){
    const f=fixture(humans),r=f.room;f.start();r.presence(0,false);f.advance(101);assert.ok(r.d.members[0].auto);
    f.cmd(0,'auto',{value:false});assert.equal(r.d.members[0].auto,false);assert.equal(r.seat(f.credentials[0].token),0);
    let n=0;while(r.d.game.phase!=='ended'&&n++<350){f.advance(101);const restored=new Room(r.serialize(),f.options);assert.deepEqual(restored.view(0),r.view(0));assertGame(r.d.game);}
    assert.equal(r.d.game.phase,'ended');assert.equal(r.d.game.players.reduce((n,p)=>n+p.score,0),0);
    assert.ok(r.view(0).review.every(x=>x.seat===0));assert.ok(JSON.stringify(r.serialize()).length<125000);
    const before=JSON.stringify(r.d.game.ledger);f.advance(101);assert.equal(JSON.stringify(r.d.game.ledger),before);
  }
});

export function socketClient(base,c){
  const ws=new WebSocket(base.replace(/^http/,'ws')+`/rooms/${c.code}/socket`),messages=[],waiters=[];
  const wait=(predicate,timeout=8000)=>new Promise((resolve,reject)=>{const found=messages.find(predicate);if(found)return resolve(found);const item={predicate,resolve,reject};item.timer=setTimeout(()=>{waiters.splice(waiters.indexOf(item),1);reject(Error('socket state timeout'));},timeout);waiters.push(item);});
  ws.on('open',()=>ws.send(JSON.stringify({type:'auth',token:c.token})));
  ws.on('error',()=>{});ws.on('message',data=>{if(data.toString()==='pong')return;const m=JSON.parse(data);messages.push(m);for(const w of [...waiters])if(w.predicate(m)){clearTimeout(w.timer);waiters.splice(waiters.indexOf(w),1);w.resolve(m);}});
  return {ws,messages,wait,send:c=>ws.send(JSON.stringify({type:'command',...c}))};
}
test('real WebSocket clients reconnect to the same persisted seat after server restart',async()=>{
  const dataDir=await mkdtemp(path.join(os.tmpdir(),'tingchuan-test-'));let server=await startServer({port:0,dataDir});let clients=[];
  try{
    let base='http://127.0.0.1:'+server.port;
    const post=async(url,name)=>{const r=await fetch(base+url,{method:'POST',body:JSON.stringify({name})});assert.equal(r.status,200);return r.json();};
    const a=await post('/rooms','房主'),b=await post(`/rooms/${a.code}/join`,'朋友');
    clients=[socketClient(base,a),socketClient(base,b)];await Promise.all(clients.map(c=>c.wait(m=>m.type==='state'&&m.members.every(x=>x.online))));
    for(let s=0;s<2;s++){clients[s].send({id:'ready-'+s,action:'ready'});await clients[s].wait(m=>m.type==='ack'&&m.id==='ready-'+s);}
    clients[0].send({id:'start',action:'start'});await Promise.all(clients.map(c=>c.wait(m=>m.type==='state'&&m.phase==='swap')));
    let state=clients[0].messages.findLast(m=>m.type==='state'),own=[...state.game.players[0].hand],rev=state.revision;
    clients[0].ws.close();await clients[1].wait(m=>m.type==='state'&&!m.members[0].online);
    const resumed=socketClient(base,a);clients.push(resumed);const recovered=await resumed.wait(m=>m.type==='state');assert.deepEqual(recovered.game.players[0].hand,own);assert.equal(recovered.seat,0);assert.ok(recovered.revision>rev);
    const command={id:'swap-once',decision:recovered.decision,action:'swap',indices:recommendSwap(own)};resumed.send(command);await resumed.wait(m=>m.type==='ack'&&m.id===command.id);
    await server.close();server=await startServer({port:0,dataDir});base='http://127.0.0.1:'+server.port;
    const again=socketClient(base,a);clients.push(again);const afterRestart=await again.wait(m=>m.type==='state');assert.equal(afterRestart.submitted,true);assert.deepEqual(afterRestart.game.players[0].hand,own);
    again.send(command);const ack=await again.wait(m=>m.type==='ack'&&m.id===command.id);assert.equal(ack.duplicate,true);
    const attacker=socketClient(base,{...a,token:'0'.repeat(64)});clients.push(attacker);const close=await new Promise(resolve=>attacker.ws.on('close',code=>resolve(code)));assert.equal(close,1008);assert.equal(attacker.messages.filter(m=>m.type==='state').length,0);
  }finally{for(const c of clients)c.ws.terminate();await server.close();}
});
