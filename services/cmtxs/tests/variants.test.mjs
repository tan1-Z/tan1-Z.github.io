import test from 'node:test';
import assert from 'node:assert/strict';
import {Room} from '../room.js';
import {RULES,JOKER,createGame,restoreGame,normalizeRules,exchangeTiles,setAllQue,recommendSwap,recommendQue,isWinning,scoreHand,shanten,legalDiscards,discard,analyze,observe,responseActions,resolveResponses,selfActions,selfAction,chooseSelfAction,chooseResponse,assertGame,meldTiles,clearQue,waits,finish} from '../../../source/cmtxs/engine.js';
const wild={...RULES,joker:true};
function fixture({hand=[],melds=[],others={},discarded=null,phase='discard',top=17,drawn=null}={}){
  const g=createGame(42,null,wild),pool=Array.from({length:108},(_,i)=>Math.floor(i/4));
  const take=t=>{const i=pool.indexOf(t);assert.notEqual(i,-1,'physical tile '+t);return pool.splice(i,1)[0];};
  g.players.forEach((p,s)=>{p.hand=[];p.melds=[];p.discards=[];p.que=s===0?2:0;});
  g.players[0].hand=hand.map(take);g.players[0].melds=structuredClone(melds);melds.flatMap(meldTiles).forEach(take);
  if(discarded!=null)g.players[3].discards=[take(discarded)];
  for(const [s,h] of Object.entries(others))g.players[s].hand=h.map(take);
  const last=take(top);for(let s=1;s<4;s++)while(g.players[s].hand.length<13)g.players[s].hand.push(pool.shift());
  g.wall=[...pool,last];g.phase=phase;g.turn=phase==='respond'?3:0;g.drawn=drawn;g.last=discarded==null?null:{tile:discarded,from:3,rob:false};assertGame(g);return g;
}
test('four independent rule combinations deal correctly and old saves keep their rules',()=>{
  for(const swap of [true,false])for(const joker of [true,false]){
    const g=createGame(12,null,{swap,joker});assert.equal(g.phase,swap?'swap':'que');assert.deepEqual(g.players.map(p=>p.hand.length),[14,13,13,13]);assertGame(g);
    if(swap)exchangeTiles(g,g.players.map(p=>recommendSwap(p.hand,g.rules)));else assert.throws(()=>exchangeTiles(g,[[0,1,2]]));
    setAllQue(g,g.players.map(p=>recommendQue(p.hand,g.rules)));assertGame(g);
    assert.deepEqual(restoreGame(JSON.parse(JSON.stringify(g))).rules,normalizeRules({swap,joker}));
  }
  const old=createGame(1);delete old.rules;assert.equal(restoreGame(old).rules.joker,false);assert.equal(old.rules.swap,true);
});
test('joker is exempt from dingque but cannot be discarded; other missing tiles block claims and wins',()=>{
  const g=fixture({hand:[0,1,2,3,4,5,9,10,11,12,13,14,18,19],drawn:19});
  assert.deepEqual(legalDiscards(g),[19]);assert.equal(clearQue(g.players[0],wild),false);assert.throws(()=>discard(g,18),/不能打出/);
  assert.equal(isWinning(g.players[0].hand,[],2,wild),false);
  g.players[0].hand.pop();assert.equal(clearQue(g.players[0],wild),true);assert.equal(legalDiscards(g).includes(18),false);
  assert.equal(analyze(observe(g,0)).some(a=>a.tile===18),false);
  assert.equal(isWinning([0,1,2,3,4,5,9,10,11,12,13,14,18,18],[],2,wild),true);
  assert.equal(isWinning([0,1,2,3,4,5,9,10,11,12,13,14,18,18],[],2,RULES),false);
});
test('joker supports pairs, lower sequence positions, open hands and best fan assignment',()=>{
  const hands=[[1,2,18,3,4,5,9,10,11,12,13,14,16,16],[0,0,1,1,2,2,3,3,4,4,18,18,18,18],[0,1,2,3,4,5,9,10,11,18,18]];
  hands.forEach((h,i)=>assert.equal(isWinning(h,i===2?[{type:'pong',tile:8}]:[],2,wild),true));
  const s=scoreHand(hands[1],[],{rules:wild,que:2,self:true});assert.equal(s.points,32);assert.equal(s.substitutions.length,4);assert.ok(s.tags.some(t=>t.name==='七对'));assert.ok(!s.tags.some(t=>t.name==='无鸡'));
});
test('no-chicken bonus includes exposed physical jokers and respects five-fan cap',()=>{
  const hand=[0,1,2,3,4,5,9,10,11,12,12];
  for(const tiles of [[8,8,8],[8,8,18]]){
    const s=scoreHand(hand,[{type:'pong',tile:8,tiles}],{rules:wild,que:2});assert.equal(s.tags.some(t=>t.name==='无鸡'),!tiles.includes(18));
  }
  const h=[0,0,0,0,1,1,2,2,3,3,4,4,5,5],s=scoreHand(h,[],{rules:wild,self:true});assert.equal(s.fan,5);assert.equal(s.rawFan,7);assert.equal(s.tags.find(t=>t.name==='无鸡').fan,1);
});
test('same-suit exchange permits physical jokers and conserves all tiles',()=>{
  // A dedicated complete pool fixture provides an actual joker packet.
  const f=fixture({hand:[18,19,20,0,1,2,3,4,5,9,10,11,12,13]});f.phase='swap';
  exchangeTiles(f,f.players.map((p,s)=>s?recommendSwap(p.hand,wild):[0,1,2]));assertGame(f);assert.ok(f.players[f.swapDirection].hand.includes(18));
});
test('exhaustive draw awards the no-chicken bonus when a natural winning tile gives the highest ready-hand payment',()=>{
  const g=createGame(1,null,wild);g.wall=[];g.phase='discard';
  g.players[0].hand=[0,1,2,3,4,5,9,10,11,12,13,14,16];g.players[0].que=2;
  g.players[1].hand=[0,0,0,1,2,4,6,8,9,11,13,15,17];g.players[1].que=2;
  g.players[2].won=g.players[3].won=true;
  assert.deepEqual(waits(g.players[0],wild),[16,18]);assert.equal(waits(g.players[1],wild).length,0);
  finish(g);assert.equal(g.ledger.find(x=>x.reason==='查大叫').amount,2);assert.equal(g.players[0].score,2);
});
test('a joker pong records actual tiles and cannot reclaim when promoted',()=>{
  const g=fixture({hand:[8,18,0,1,2,3,4,5,9,10,11,12,13],discarded:8,phase:'respond'});
  assert.ok(responseActions(g,0).includes('pong:1'));resolveResponses(g,['pong:1','pass','pass','pass']);assertGame(g);
  assert.deepEqual(meldTiles(g.players[0].melds[0]),[8,8,18]);assert.equal(selfActions(g,0).some(a=>a.type==='reclaim'),false);
  const f=fixture({hand:[8,0,1,2,3,4,5,9,10,11,12],melds:[{type:'pong',tile:8,from:3,tiles:[8,8,18]}],drawn:8,top:16});
  selfAction(f,'addkong',8);assert.equal(f.phase,'respond');resolveResponses(f,['pass','pass','pass','pass']);assertGame(f);
  assert.equal(f.players[0].melds[0].type,'addkong');assert.deepEqual(meldTiles(f.players[0].melds[0]),[8,8,18,8]);assert.equal(f.players[0].hand.includes(18),false);
});
test('a joker kong can exchange a newly drawn matching tile for exactly one joker without extra tax or draw',()=>{
  const g=fixture({hand:[8,8,18,0,1,2,3,4,5,9,10,11,12],discarded:8,phase:'respond',top:8});
  assert.ok(responseActions(g,0).includes('kong:1'));resolveResponses(g,['kong:1','pass','pass','pass']);assertGame(g);
  assert.ok(selfActions(g,0).some(a=>a.type==='reclaim'&&a.tile===8));const wall=g.wall.length,ledger=structuredClone(g.ledger),length=g.players[0].hand.length;
  selfAction(g,'reclaim',8);assertGame(g);assert.deepEqual(meldTiles(g.players[0].melds[0]),[8,8,8,8]);assert.ok(g.players[0].hand.includes(18));assert.equal(g.players[0].hand.length,length);assert.equal(g.wall.length,wall);assert.deepEqual(g.ledger,ledger);assert.equal(selfActions(g,0).some(a=>a.type==='reclaim'),false);
  assert.throws(()=>selfAction(g,'reclaim',8));
});
test('concealed and added joker kongs retain physical jokers and reject forged joker counts atomically',()=>{
  const g=fixture({hand:[8,8,18,18,0,1,2,3,4,5,9,10,11,12],drawn:12});
  assert.ok(selfActions(g,0).some(a=>a.type==='ankong'&&a.tile===8&&a.jokers===2));const before=JSON.stringify(g);assert.throws(()=>selfAction(g,'ankong',8,0,3));assert.equal(JSON.stringify(g),before);
  selfAction(g,'ankong',8,0,2);assertGame(g);assert.deepEqual(meldTiles(g.players[0].melds[0]),[8,8,18,18]);
  const f=fixture({hand:[18,0,1,2,3,4,5,9,10,11,12],melds:[{type:'pong',tile:8,from:3,tiles:[8,8,8]}],drawn:18});
  selfAction(f,'addkong',8,0,1);resolveResponses(f,['pass','pass','pass','pass']);assertGame(f);assert.deepEqual(meldTiles(f.players[0].melds[0]),[8,8,8,18]);
});
test('robbing a joker added kong uses the declared face, retains the original pong and grants no no-chicken bonus',()=>{
  const g=fixture({hand:[18,0,1,2,3,4,5,9,10,11,12],melds:[{type:'pong',tile:8,from:3,tiles:[8,8,8]}],others:{1:[0,1,2,3,4,5,9,10,11,12,13,14,8]},drawn:18});g.players[1].que=2;
  selfAction(g,'addkong',8,0,1);assert.ok(responseActions(g,1).includes('hu'));resolveResponses(g,['pass','hu','pass','pass']);assertGame(g);
  assert.equal(g.players[0].melds[0].type,'pong');assert.equal(g.players[1].won,true);assert.equal(g.players[1].win.tags.some(t=>t.name==='无鸡'),false);assert.ok(g.players[1].win.tags.some(t=>t.name==='抢杠胡'));assert.ok(!g.ledger.some(x=>x.kind==='kong'));
});
test('joker pass-hu locks and no-chicken settlement use actual capped fan',()=>{
  const g=fixture({hand:[0,1,2,3,4,5,9,10,11,12,13,14,18],discarded:8,phase:'respond'});
  assert.ok(responseActions(g).includes('hu'));const score=scoreHand([...g.players[0].hand,8],[],{rules:wild,que:2});g.players[0].passedHuFan=score.fan;
  assert.equal(responseActions(g).includes('hu'),false);assert.throws(()=>resolveResponses(g,['hu','pass','pass','pass']),/过手胡/);
  const p={hand:[0,1,2,3,4,5,9,10,11,12,13,14,18],melds:[],que:2};assert.ok(waits(p,wild).includes(18));assert.equal(clearQue(p,wild),true);
});
test('wildcard shanten matches independent exhaustive substitutions for one and two jokers',()=>{
  let seed=723;const rand=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/2**32);
  for(const open of [0,1,3])for(const j of [1,2])for(const len of [13-open*3,14-open*3])for(let k=0;k<8;k++){
    const natural=[];while(natural.length<len-j){const t=Math.floor(rand()*27);if(t!==18&&natural.filter(x=>x===t).length<4)natural.push(t);}
    let best=8;const walk=(a,n,start)=>{if(!n){best=Math.min(best,shanten(a,open));return;}for(let t=start;t<27;t++)if(a.filter(x=>x===t).length<4)walk([...a,t],n-1,t);};walk(natural,j,0);
    assert.equal(shanten([...natural,...Array(j).fill(18)],open,wild),best,JSON.stringify({open,j,natural}));
  }
});
test('host-only rules clear readiness, persist reconnects and cannot alter an active game',()=>{
  const r=new Room(),a=r.init('ABCD2345','房主'),b=r.join('牌友');let id=0;const cmd=(seat,action,extra={})=>r.command(seat,{id:String(++id),decision:r.d.decision,action,...extra});
  cmd(0,'ready');cmd(1,'ready');assert.throws(()=>cmd(1,'configure',{rules:{joker:true,swap:false}}),/房主/);cmd(0,'configure',{rules:{joker:true,swap:false}});assert.ok(r.d.members.filter(Boolean).every(m=>!m.ready));
  const copy=new Room(r.serialize());assert.equal(copy.seat(a.token),0);assert.equal(copy.seat(b.token),1);assert.equal(copy.view(1).rules.joker,true);assert.equal(copy.view(1).rules.swap,false);
  cmd(0,'ready');cmd(1,'ready');cmd(0,'start');assert.equal(r.d.game.phase,'que');assert.equal(r.d.game.rules.joker,true);assert.throws(()=>cmd(0,'configure',{rules:{joker:false,swap:true}}),/已开始/);
});
test('all four variants finish mixed online games without illegal joker discards or hidden-tile leakage',()=>{
  for(const swap of [true,false])for(const joker of [true,false])for(let seed=1;seed<=3;seed++){
    let time=0,random=seed;const options={now:()=>time,timeout:10,botDelay:1,random:()=>((random=(Math.imul(random,1664525)+1013904223)>>>0)/2**32)};
    let r=new Room(null,options);r.init('ABCD2345','房主');r.join('牌友');let id=0;
    const command=(s,action,extra={})=>r.command(s,{id:String(++id),decision:r.d.decision,action,...extra});command(0,'configure',{rules:{swap,joker}});command(0,'ready');command(1,'ready');command(0,'start');
    let steps=0;while(r.d.game.phase!=='ended'&&steps++<800){time+=11;r.tick();assertGame(r.d.game);if(steps===12)r=new Room(r.serialize(),options);}
    assert.equal(r.d.game.phase,'ended',JSON.stringify({swap,joker,seed}));assert.ok(r.view(0).review.every(x=>analyze(x.obs).some(a=>a.tile===x.chosen)));
    if(joker)assert.ok(r.d.game.players.every(p=>!p.discards.includes(JOKER)));
  }
});
