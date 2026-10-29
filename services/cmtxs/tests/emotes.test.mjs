import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {WebSocket} from 'ws';
import {EMOTES,EMOTE_COOLDOWN_MS,EMOTE_DURATION_MS,isEmoteId,EmoteFeed} from '../../../source/cmtxs/emotes.js';
import {emotePanel} from '../../../source/cmtxs/emotes-ui.js';
import {Room} from '../room.js';
import {startServer} from '../local-server.mjs';
import {recommendSwap,isCompatibleVersion} from '../../../source/cmtxs/engine.js';

function fixture(){let time=100000;const options={now:()=>time,timeout:30000,botDelay:700},r=new Room(null,options);r.init('EMOT2345','房主');r.join('朋友');let id=0;const cmd=(seat,action,extra={})=>r.command(seat,{id:'game-'+(++id),decision:r.d.decision,action,...extra});return {r,options,cmd,start(){cmd(0,'ready');cmd(1,'ready');cmd(0,'start');},advance(ms){time+=ms;}};}
const packet=(id='one',sticker='steady')=>({id,sticker});
test('nine curated images exist as compact transparent WebP assets',async()=>{
  assert.equal(EMOTES.length,9);assert.equal(new Set(EMOTES.map(e=>e.id)).size,9);assert.equal(EMOTES.filter(e=>e.group==='牌友').length,6);assert.equal(EMOTES.filter(e=>e.group==='原创').length,3);
  for(const e of EMOTES){const file=new URL('../../../source/cmtxs/assets/emotes/'+e.id+'.webp',import.meta.url),data=await readFile(file);assert.equal(data.toString('ascii',0,4),'RIFF');assert.equal(data.toString('ascii',8,12),'WEBP');assert.ok(data.length<80000);}
  for(const bad of ['../evil','https://a/img.jpg','<script>','',null,{}])assert.equal(isEmoteId(bad),false);
  assert.equal(isCompatibleVersion('0.5.0'),true);
});
test('sending never mutates the game, pending choices, deadline, revision or readiness',()=>{
  const f=fixture();f.start();const before=f.r.serialize(),result=f.r.sendEmote(1,packet());const after=f.r.serialize();
  delete before.emoteLimits;delete after.emoteLimits;assert.deepEqual(after,before);assert.equal(result.event.seat,1);assert.equal(result.event.expiresAt-result.event.at,EMOTE_DURATION_MS);
  assert.deepEqual(Object.keys(result.event).sort(),['at','expiresAt','id','seat','sticker','type']);
  assert.equal(f.r.view(0).emoteLimits,undefined);assert.equal(f.r.view(0).emotes,undefined);
  f.cmd(0,'swap',{indices:recommendSwap(f.r.d.game.players[0].hand)});assert.equal(f.r.view(0).submitted,true);
});
test('server enforces a per-seat cooldown across reconnect and restart',()=>{
  const f=fixture();f.start();f.r.sendEmote(0,packet());assert.throws(()=>f.r.sendEmote(0,packet('two')),/太快/);
  assert.equal(f.r.sendEmote(1,packet('one')).event.seat,1);const restored=new Room(f.r.serialize(),f.options);assert.throws(()=>restored.sendEmote(0,packet('three')),/太快/);
  f.advance(EMOTE_COOLDOWN_MS-1);assert.throws(()=>restored.sendEmote(0,packet('three')),/太快/);f.advance(1);assert.ok(restored.sendEmote(0,packet('three')).event);
});
test('duplicate sends acknowledge once without replaying or extending the cooldown',()=>{
  const f=fixture();f.start();f.r.sendEmote(0,packet());const saved=f.r.serialize();f.advance(1000);assert.deepEqual(f.r.sendEmote(0,packet()),{id:'one',duplicate:true});assert.deepEqual(f.r.serialize(),saved);
  f.advance(2000);assert.ok(f.r.sendEmote(0,packet('two')).event);
});
test('unknown images, malformed IDs, missing seats and pregame sends are rejected atomically',()=>{
  const f=fixture();const before=f.r.serialize();assert.throws(()=>f.r.sendEmote(0,packet()),/开局/);f.start();
  const afterStart=f.r.serialize();for(const bad of [packet('x','https://evil/x'),packet('x','../thinking'),packet('<script>'),null])assert.throws(()=>f.r.sendEmote(0,bad),/无效/);
  assert.throws(()=>f.r.sendEmote(3,packet()),/加入/);assert.deepEqual(f.r.serialize(),afterStart);assert.ok(before.emoteLimits.every(x=>x.at===null));
});
test('won players and humans using takeover can send without changing their game status',()=>{
  const f=fixture();f.start();f.cmd(1,'auto',{value:true});const state=f.r.serialize();assert.equal(f.r.sendEmote(1,packet()).event.seat,1);assert.equal(f.r.d.members[1].auto,true);assert.deepEqual(f.r.d.game,state.game);
  f.r.d.game.players[0].won=true;assert.equal(f.r.sendEmote(0,packet('winner','kong-flower')).event.seat,0);
});
test('old persisted rooms gain safe empty limits and retain seat credentials',()=>{
  const f=fixture();f.start();const old=f.r.serialize();delete old.emoteLimits;const r=new Room(old,f.options);assert.equal(r.seat(old.members[1].token),1);assert.equal(r.sendEmote(1,packet()).event.seat,1);
});
test('the display inbox expires, deduplicates and replaces one sticker per seat',()=>{
  let now=100000;const feed=new EmoteFeed({now:()=>now}),e={...packet(),seat:1,at:now,expiresAt:now+EMOTE_DURATION_MS};assert.equal(feed.receive(e),true);assert.equal(feed.receive(e),false);assert.equal(feed.current().length,1);
  now+=1000;assert.equal(feed.receive({...e,id:'two',sticker:'lucky',at:now,expiresAt:now+EMOTE_DURATION_MS}),true);assert.equal(feed.current()[0].sticker,'lucky');
  assert.equal(feed.receive({...e,id:'three',seat:2}),true);assert.equal(feed.current().length,2);now+=EMOTE_DURATION_MS;assert.equal(feed.current().length,0);
});
test('mute immediately removes stickers and never replays them when unmuted',()=>{
  const now=100000,feed=new EmoteFeed({now:()=>now}),e={...packet(),seat:0,at:now,expiresAt:now+4200};feed.receive(e);feed.setMuted(true);assert.equal(feed.current().length,0);assert.equal(feed.receive({...e,id:'muted'}),false);feed.setMuted(false);assert.equal(feed.receive({...e,id:'muted'}),false);assert.equal(feed.current().length,0);
});
test('the inbox honors clock skew and rejects late or forged image events',()=>{
  const now=100000,feed=new EmoteFeed({now:()=>now}),e={...packet(),seat:0,at:now+60000,expiresAt:now+64200};assert.equal(feed.receive(e,60000),true);feed.clear();
  for(const bad of [{...e,seat:4},{...e,sticker:'../x'},{...e,expiresAt:now+999999},{...e,at:NaN},{...e,id:'bad id'}])assert.equal(feed.receive(bad,60000),false);
  assert.equal(feed.receive({...e,at:now-5000,expiresAt:now-800}),false);
});
test('the picker exposes all nine images, an independent mute control and cooldown feedback',()=>{
  const html=emotePanel({open:true,cooldown:1500,muted:true});assert.equal((html.match(/data-emote-id=/g)||[]).length,9);assert.equal((html.match(/ disabled/g)||[]).length,9);assert.match(html,/2s/);assert.match(html,/恢复显示同桌表情/);assert.match(html,/role="dialog"/);assert.doesNotMatch(emotePanel(),/emote-choice/);
});

function client(base,credential){
  const ws=new WebSocket(base.replace('http:','ws:')+`/rooms/${credential.code}/socket`),messages=[],waiters=[];
  const wait=fn=>new Promise((resolve,reject)=>{const previous=messages.find(fn);if(previous)return resolve(previous);const item={fn,resolve,timer:setTimeout(()=>reject(Error('emote socket timeout')),8000)};waiters.push(item);});
  ws.on('open',()=>ws.send(JSON.stringify({type:'auth',token:credential.token})));ws.on('error',()=>{});ws.on('message',b=>{const m=JSON.parse(b.toString());messages.push(m);for(const w of [...waiters])if(w.fn(m)){clearTimeout(w.timer);waiters.splice(waiters.indexOf(w),1);w.resolve(m);}});
  return {ws,messages,wait,send:m=>ws.send(JSON.stringify(m))};
}
test('real sockets broadcast authenticated seats, keep game commands independent and do not replay after reconnect',async()=>{
  const server=await startServer({port:0,timeout:60000,botDelay:60000}),base='http://127.0.0.1:'+server.port,clients=[];
  try{
    const post=async(url,name)=>(await fetch(base+url,{method:'POST',body:JSON.stringify({name})})).json();const a=await post('/rooms','主'),b=await post(`/rooms/${a.code}/join`,'友');
    clients.push(client(base,a),client(base,b));await Promise.all(clients.map(c=>c.wait(m=>m.type==='state'&&m.members.every(p=>p.online))));
    for(let s=0;s<2;s++){clients[s].send({type:'command',id:'ready-'+s,action:'ready'});await clients[s].wait(m=>m.type==='ack'&&m.id==='ready-'+s);}
    clients[0].send({type:'command',id:'start',action:'start'});await Promise.all(clients.map(c=>c.wait(m=>m.type==='state'&&m.phase==='swap')));
    const before=clients[0].messages.findLast(m=>m.type==='state'),event={type:'emote',id:'network-one',sticker:'chicken',seat:3};clients[0].send(event);
    const received=await clients[1].wait(m=>m.type==='emote');assert.equal(received.seat,0);assert.equal(received.sticker,'chicken');assert.equal(received.id,event.id);
    clients[0].send(event);assert.equal((await clients[0].wait(m=>m.type==='emote-ack'&&m.duplicate)).duplicate,true);
    clients[0].send({...event,id:'too-soon'});assert.match((await clients[0].wait(m=>m.type==='emote-error')).message,/太快/);
    clients[0].send({type:'command',id:'swap',decision:before.decision,action:'swap',indices:recommendSwap(before.game.players[0].hand)});await clients[0].wait(m=>m.type==='ack'&&m.id==='swap');
    const current=clients[0].messages.findLast(m=>m.type==='state');assert.equal(current.deadline,before.deadline);assert.equal(current.decision,before.decision);assert.equal(current.submitted,true);
    clients[0].ws.close();await clients[1].wait(m=>m.type==='state'&&!m.members[0].online);const resumed=client(base,a);clients.push(resumed);const state=await resumed.wait(m=>m.type==='state');assert.equal(state.submitted,true);assert.equal(resumed.messages.some(m=>m.type==='emote'),false);assert.equal(state.emotes,undefined);
    assert.equal(clients[1].messages.filter(m=>m.type==='emote').length,1);
  }finally{clients.forEach(c=>c.ws.terminate());await server.close();}
});
