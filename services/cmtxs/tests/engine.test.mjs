import test from 'node:test';
import assert from 'node:assert/strict';
import { createGame,swapTiles,setQue,recommendSwap,recommendQue,assertGame,aiStep,analyze,observe,discard,selfActions,selfAction,resolveResponse,isWinning,shanten,scoreHand,waits,finish } from '../../../source/cmtxs/engine.js';
test('standard, seven pairs, open melds and dingque legality',()=>{
  const h=[0,1,2,3,4,5,9,10,11,18,18,18,20,20];
  assert.equal(isWinning(h),true);assert.equal(shanten(h),-1);
  assert.equal(isWinning(h,[],2),false);
  assert.equal(isWinning([0,0,0,0,1,1,2,2,3,3,4,4,5,5]),true);
  assert.equal(isWinning([0,0,1,1,2,2,3,3,4,4,5,5,6,7]),false);
  assert.equal(isWinning([0,1,2,9,10,11,18,18,18,20,20],[{type:'pong',tile:3}]),true);
});
test('fan, roots, self-draw and five-fan cap',()=>{
  const flat=scoreHand([0,1,2,3,4,5,9,10,11,18,18,18,20,20],[],{self:true});
  assert.equal(flat.fan,1);assert.equal(flat.points,2);
  const dragon=scoreHand([0,0,0,0,1,1,2,2,3,3,4,4,5,5],[],{self:true});
  assert.equal(dragon.fan,5);assert.equal(dragon.rawFan,6);assert.equal(dragon.points,32);
});
test('exchange is simultaneous, same-suit and conservative',()=>{
  const g=createGame(15);assertGame(g);
  assert.throws(()=>swapTiles(g,[0,0,0]));
  swapTiles(g,recommendSwap(g.players[0].hand));assertGame(g);
  assert.equal(g.players[0].hand.length,14);assert.equal(g.players[1].hand.length,13);
  setQue(g,recommendQue(g.players[0].hand));assertGame(g);
});
test('evaluation sees only the player observation and obeys dingque',()=>{
  const g=createGame(22);swapTiles(g,recommendSwap(g.players[0].hand));setQue(g,0);
  const obs=observe(g,0),a=analyze(obs);assert.ok(a.length);assert.ok(a.every(x=>x.tile<9));
  const before=JSON.stringify(a);g.players[1].hand.reverse();assert.equal(JSON.stringify(analyze(observe(g,0))),before);
});
test('exhaustive draw settles flower pigs, ready hands and kong refunds',()=>{
  const g=createGame(1);g.phase='discard';g.wall=[];
  g.players[0].que=2;g.players[0].hand=[0,1,2,3,4,5,9,10,11,12,13,14,16];
  g.players[1].que=2;g.players[1].hand=[0,0,0,1,2,4,6,8,9,11,13,15,17];
  g.players[2].que=2;g.players[2].hand=[0,1,2,3,4,5,9,10,11,12,13,14,18];
  g.players[3].won=true;g.players[3].que=2;g.players[3].hand=[0,1,2,3,4,5,9,10,11,12,12,12,16,16];
  g.ledger.push({from:0,to:1,amount:2,kind:'kong',reason:'直杠',refunded:false});g.players[0].score=-2;g.players[1].score=2;
  assert.ok(waits(g.players[0]).length);assert.equal(waits(g.players[1]).length,0);
  finish(g);assert.equal(g.phase,'ended');assert.ok(g.ledger.some(x=>x.reason==='查花猪'));assert.ok(g.ledger.some(x=>x.reason==='查大叫'));assert.ok(g.ledger.some(x=>x.kind==='refund'));
  assert.equal(g.players.reduce((n,p)=>n+p.score,0),0);
});
test('two players can ron the same discard without duplicating a physical tile',()=>{
  const g=createGame(31);g.phase='respond';g.turn=3;g.last={tile:8,from:3,rob:false};
  const h=[0,1,2,3,4,5,9,10,11,12,13,14,8];
  g.players[0].que=2;g.players[0].hand=[...h];g.players[1].que=2;g.players[1].hand=[...h];
  g.players[2].que=0;g.players[2].hand=[0,9,10,11,12,13,14,15,16,17,18,19,20];
  g.players[3].discards=[8];g.decision={tile:8,from:3,actions:['hu']};
  resolveResponse(g,'hu');assert.deepEqual(g.winners,[0,1]);
  assert.equal(g.players[0].hand.length,13);assert.equal(g.players[1].hand.length,13);
  assert.equal(g.players[3].discards.filter(t=>t===8).length,1);
  assert.equal(g.players.reduce((n,p)=>n+p.score,0),0);
});
test('robbed added kong keeps the original pong and charges no kong tax',()=>{
  const g=createGame(32);g.phase='respond';g.turn=3;g.last={tile:8,from:3,rob:true};
  g.players[0].que=2;g.players[0].hand=[0,1,2,3,4,5,9,10,11,12,13,14,8];
  g.players[1].que=0;g.players[1].hand=[0,9,10,11,12,13,14,15,16,17,18,19,20];
  g.players[2].que=0;g.players[2].hand=[0,9,10,11,12,13,14,15,16,17,18,19,20];
  g.players[3].que=2;g.players[3].hand=[8,0,1,2,3,4,5,9,10,11,12];g.players[3].melds=[{type:'pong',tile:8,from:1}];
  g.decision={tile:8,from:3,actions:['hu']};resolveResponse(g,'hu');
  assert.equal(g.players[0].won,true);assert.equal(g.players[3].melds[0].type,'pong');
  assert.ok(g.players[0].win.tags.some(t=>t.name==='抢杠胡'));assert.equal(g.ledger.some(l=>l.kind==='kong'),false);
});
test('a pong response does not turn into an illegal self draw',()=>{
  const g=createGame(33);g.phase='discard';g.turn=0;g.drawn=null;g.players[0].que=2;
  g.players[0].hand=[0,1,2,3,4,5,9,10,11,12,12];g.players[0].melds=[{type:'pong',tile:8,from:1}];
  assert.equal(isWinning(g.players[0].hand,g.players[0].melds,2),true);
  assert.equal(selfActions(g).some(a=>a.type==='hu'),false);
});
test('complete seeded games preserve all 108 tiles and zero-sum accounting',()=>{
  for(let seed=1;seed<=16;seed++){
    const g=createGame(seed);swapTiles(g,recommendSwap(g.players[0].hand));setQue(g,recommendQue(g.players[0].hand));
    let steps=0;
    while(g.phase!=='ended'&&steps++<500){
      if(g.phase==='respond')resolveResponse(g,g.decision?.actions.includes('hu')?'hu':'pass');
      else if(g.phase==='discard'&&g.turn===0){
        const actions=selfActions(g),act=actions.find(a=>a.type==='hu')||actions.find(a=>a.type.includes('kong'));
        if(act)selfAction(g,act.type,act.tile);else discard(g,analyze(observe(g,0))[0].tile);
      }else aiStep(g);
      assertGame(g);
    }
    assert.equal(g.phase,'ended',`seed ${seed} did not finish`);
    assert.ok(g.events.filter(e=>e.review).every(e=>e.review.analysis.some(a=>a.tile===e.review.chosen)));
  }
});
