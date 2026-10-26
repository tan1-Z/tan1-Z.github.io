import test from 'node:test';
import assert from 'node:assert/strict';
import { createGame,sorted,discard,resolveResponse,assertGame,restoreGame,responseActions,selfActions,scoreHand } from '../../../source/cmtxs/engine.js';

const normalWait=[0,1,2,3,4,5,9,10,11,12,13,14,15];
// Every fixture has all 108 physical tiles. AI seats retain their missing suit,
// so an unrelated computer claim cannot change the intended response sequence.
function fixture({hand=normalWait,forced=[[9],[12],[15]],draws=[20,21,9],aiQue=1}={}){
  const g=createGame(101),pool=Array.from({length:108},(_,i)=>Math.floor(i/4));
  const take=t=>{const i=pool.indexOf(t);assert.notEqual(i,-1,`excess ${t}`);pool.splice(i,1);return t;};
  g.players.forEach(p=>{p.hand=[];p.melds=[];p.discards=[];p.que=p.seat===0?2:aiQue;});
  g.players[0].hand=hand.map(take);
  for(let s=1;s<4;s++)g.players[s].hand=forced[s-1].map(take);
  const reserved=draws.map(take);
  for(let s=1;s<4;s++)while(g.players[s].hand.length<(s===1?14:13))g.players[s].hand.push(pool.shift());
  for(const p of g.players)p.hand=sorted(p.hand);
  g.wall=[...pool,...reserved.reverse()];g.turn=1;g.phase='discard';g.drawn=g.players[1].hand.at(-1);
  assertGame(g);return g;
}

test('passing a ron blocks a different winning tile at the same fan',()=>{
  const g=fixture();discard(g,9,1);assert.ok(g.decision?.actions.includes('hu'));
  resolveResponse(g,'pass');assert.equal(g.turn,2);
  discard(g,12,2);
  assert.equal(g.decision?.actions.includes('hu')??false,false);
  const before=JSON.stringify(g);assert.throws(()=>resolveResponse(g,'hu'),/不合法|过手/);
  assert.equal(JSON.stringify(g),before);assertGame(g);
});

test('higher fan remains claimable before the next own draw',()=>{
  const g=fixture({hand:[0,0,0,1,2,9,10,11,12,13,14,15,15],forced:[[3],[0],[3]],draws:[20,21,22],aiQue:0});
  discard(g,3,1);assert.ok(g.decision?.actions.includes('hu'));resolveResponse(g,'pass');
  discard(g,0,2);assert.ok(g.decision?.actions.includes('hu'));
  resolveResponse(g,'hu');assert.equal(g.players[0].won,true);assert.equal(g.players[0].win.fan,1);assertGame(g);
});

test('same winning tile is also blocked; only the owner draw clears the lock',()=>{
  const g=fixture({forced:[[9],[9],[15]]});
  discard(g,9,1);resolveResponse(g,'pass');assert.equal(g.players[0].passedHuFan,0);
  // Seat 2 has drawn; this does not clear seat 0's refusal.
  assert.equal(g.turn,2);discard(g,9,2);assert.ok(!g.decision?.actions.includes('hu'));
  resolveResponse(g,'pass');assert.equal(g.players[0].passedHuFan,0);
  discard(g,15,3);assert.ok(!g.decision?.actions.includes('hu'));resolveResponse(g,'pass');
  assert.equal(g.turn,0);assert.equal(g.players[0].passedHuFan,null);
  assert.ok(selfActions(g).some(a=>a.type==='hu'));assertGame(g);
});

test('passing a higher fan raises the threshold and lower fan stays blocked',()=>{
  const g=fixture({hand:[0,0,0,1,2,9,10,11,12,13,14,15,15],forced:[[3],[0],[3]],draws:[20,21,22],aiQue:0});
  discard(g,3,1);resolveResponse(g,'pass');assert.equal(g.players[0].passedHuFan,0);
  discard(g,0,2);resolveResponse(g,'pass');assert.equal(g.players[0].passedHuFan,1);
  discard(g,3,3);assert.ok(!g.decision?.actions.includes('hu'));
  assert.deepEqual(g.events.filter(e=>e.type==='pass-hu').map(e=>e.fan),[0,1]);assertGame(g);
});

test('choosing pong instead of an available ron records and retains the refusal',()=>{
  const g=fixture({hand:[0,1,2,3,4,5,9,9,10,11,12,15,15],forced:[[9,15],[12],[15]],draws:[20,21,22]});
  discard(g,9,1);assert.ok(g.decision.actions.includes('hu'));assert.ok(g.decision.actions.includes('pong'));
  resolveResponse(g,'pong');assert.equal(g.turn,0);assert.equal(g.players[0].passedHuFan,0);
  discard(g,15,0);resolveResponse(g,'pass');discard(g,15,1);
  assert.ok(!g.decision?.actions.includes('hu'));assert.equal(g.players[0].passedHuFan,0);assertGame(g);
});

test('a kong replacement draw clears the refusal after choosing kong over ron',()=>{
  const g=fixture({hand:[0,0,0,1,2,9,10,11,12,13,14,15,15],forced:[[0],[12],[15]],draws:[20,21,22]});
  discard(g,0,1);assert.ok(g.decision.actions.includes('hu'));assert.ok(g.decision.actions.includes('kong'));
  resolveResponse(g,'kong');assert.equal(g.players[0].passedHuFan,null);assert.equal(g.afterKong,true);
  const declined=g.events.find(e=>e.type==='pass-hu');assert.equal(declined.fan,1);
  assert.equal(declined.state.players[0].passedHuFan,1);assertGame(g);
});

test('serialization and stale stored buttons cannot bypass the refusal',()=>{
  const g=fixture();discard(g,9,1);resolveResponse(g,'pass');discard(g,12,2);
  const restored=JSON.parse(JSON.stringify(g));restored.decision={...restored.last,actions:['hu']};
  restoreGame(restored);assert.equal(restored.players[0].passedHuFan,0);assert.equal(restored.decision,null);
  assert.throws(()=>resolveResponse(restored,'hu'),/过手/);assertGame(restored);
});

test('legacy local games recover resolved refusals but not unresolved prompts',()=>{
  const g=fixture();discard(g,9,1);
  const pending=JSON.parse(JSON.stringify(g));pending.version='0.1.0';pending.players.forEach(p=>delete p.passedHuFan);
  restoreGame(pending);assert.equal(pending.players[0].passedHuFan,null);assert.ok(pending.decision.actions.includes('hu'));
  resolveResponse(g,'pass');discard(g,12,2);
  const legacy=JSON.parse(JSON.stringify(g));legacy.version='0.1.0';legacy.players.forEach(p=>delete p.passedHuFan);legacy.events=legacy.events.filter(e=>e.type!=='pass-hu');
  restoreGame(legacy);assert.equal(legacy.players[0].passedHuFan,0);assert.equal(legacy.decision,null);assertGame(legacy);
});

test('legacy restoration respects own draws, pongs, and an accepted ron',()=>{
  const g=fixture();discard(g,9,1);resolveResponse(g,'pass');discard(g,12,2);resolveResponse(g,'pass');discard(g,15,3);resolveResponse(g,'pass');
  g.players.forEach(p=>delete p.passedHuFan);g.events=g.events.filter(e=>e.type!=='pass-hu');
  restoreGame(g);assert.equal(g.players[0].passedHuFan,null);
  const pong=fixture({hand:[0,1,2,3,4,5,9,9,10,11,12,15,15],forced:[[9,15],[12],[15]],draws:[20,21,22]});
  discard(pong,9,1);resolveResponse(pong,'pong');pong.players.forEach(p=>delete p.passedHuFan);pong.events=pong.events.filter(e=>e.type!=='pass-hu');
  restoreGame(pong);assert.equal(pong.players[0].passedHuFan,0);
  const won=fixture();discard(won,9,1);resolveResponse(won,'hu');won.players.forEach(p=>delete p.passedHuFan);
  restoreGame(won);assert.equal(won.players[0].passedHuFan,null);
});

test('refusal is scoped to the seat and uses capped settlement fan, including rob-kong',()=>{
  const g=fixture();discard(g,9,1);g.players[0].passedHuFan=0;
  assert.ok(!responseActions(g,0).includes('hu'));
  const p=g.players[0];p.hand=[3];p.melds=[1,2,9,10].map(tile=>({type:'kong',tile,from:1}));p.passedHuFan=5;
  g.last={tile:3,from:1,rob:true};g.phase='respond';
  const scored=scoreHand([3,3],p.melds,{rob:true});assert.ok(scored.rawFan>5);assert.equal(scored.fan,5);
  assert.ok(!responseActions(g,0).includes('hu'));
  p.passedHuFan=null;assert.ok(responseActions(g,0).includes('hu'));
});

test('computer responders use the same refusal gate',()=>{
  const g=fixture();discard(g,9,1);
  // Move the same waiting hand and lock to the computer sitting across from the discarder.
  [g.players[0].hand,g.players[2].hand]=[g.players[2].hand,g.players[0].hand];
  g.players[2].que=2;g.players[2].passedHuFan=0;g.players[0].que=0;
  restoreGame(g);assert.ok(!responseActions(g,2).includes('hu'));
  resolveResponse(g,'pass');assert.equal(g.players[2].won,false);assertGame(g);
});

test('passing a non-winning discard never starts a refusal',()=>{
  const g=fixture({forced:[[16],[12],[15]]});discard(g,16,1);assert.ok(!g.decision?.actions.includes('hu'));
  resolveResponse(g,'pass');assert.equal(g.players[0].passedHuFan,null);
  discard(g,12,2);assert.ok(g.decision.actions.includes('hu'));assertGame(g);
});
