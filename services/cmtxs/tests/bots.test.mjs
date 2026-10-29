import test from 'node:test';
import assert from 'node:assert/strict';
import {createGame,restoreGame,setAllQue,recommendQue,exchangeTiles,recommendSwap,observe,analyze,chooseDiscard,botDiscardAnalysis,chooseResponse,resolveResponses,chooseSelfAction,selfAction,discard,assertGame,meldTiles,finish} from '../../../source/cmtxs/engine.js';
import {BOT_STYLES,DEFAULT_BOT_STYLES,normalizeBotStyles,normalizeBotStyle,rankBotDiscards,tileRisk} from '../../../source/cmtxs/bot-policy.js';
import {Room} from '../room.js';
import {botPicker} from '../../../source/cmtxs/bots-ui.js';
const styles=Object.keys(BOT_STYLES),decisionHand=[0,1,2,2,7,7,7,9,11,13,14,14,17,17];
function fixture({hand=decisionHand,melds=[],opponentMelds=[],respond=null}={}){
  const g=createGame(42,null,{swap:false}),pool=Array.from({length:108},(_,i)=>Math.floor(i/4));
  const take=t=>{const i=pool.indexOf(t);assert.ok(i>=0,'physical tile '+t);return pool.splice(i,1)[0];};
  for(const p of g.players){p.hand=[];p.melds=[];p.discards=[];p.que=p.seat?1:2;}
  g.players[0].hand=hand.map(take);g.players[0].melds=structuredClone(melds);melds.flatMap(meldTiles).forEach(take);
  g.players[1].melds=structuredClone(opponentMelds);opponentMelds.flatMap(meldTiles).forEach(take);
  if(respond!=null)g.players[3].discards=[take(respond)];
  for(let s=1;s<4;s++)while(g.players[s].hand.length<13-3*g.players[s].melds.length)g.players[s].hand.push(pool.shift());
  g.wall=pool;g.phase=respond==null?'discard':'respond';g.turn=respond==null?0:3;g.drawn=respond==null?hand.at(-1):null;g.last=respond==null?null:{tile:respond,from:3,rob:false};assertGame(g);return g;
}
test('bot settings normalize safely and legacy games retain balanced profiles',()=>{
  assert.equal(normalizeBotStyle('__proto__'),'balanced');assert.deepEqual(normalizeBotStyles(['unknown','fast','big','defensive']),['balanced','fast','big','defensive']);
  const a=createGame(1),b=createGame(1);assert.deepEqual(a.botStyles,[...DEFAULT_BOT_STYLES]);a.botStyles[1]='big';assert.equal(b.botStyles[1],'fast');
  const old=createGame(1);old.version='0.3.0';delete old.botStyles;assert.deepEqual(restoreGame(old).botStyles,Array(4).fill('balanced'));
});
test('identical visible hands produce a wider fast choice and a higher-potential big-hand choice',()=>{
  const g=fixture(),obs=observe(g,0),analysis=analyze(obs),fast=rankBotDiscards(obs,analysis,'fast')[0],big=rankBotDiscards(obs,analysis,'big')[0];
  assert.equal(fast.tile,2);assert.equal(big.tile,9);assert.equal(fast.shanten,big.shanten);assert.ok(fast.ukeire>big.ukeire);assert.ok(big.potential.points>fast.potential.points);
});
test('a defensive discard respects public missing suits and reduces estimated exposure',()=>{
  const g=fixture({opponentMelds:[0,3,6].map(tile=>({type:'pong',tile,from:2}))});
  g.botStyles[0]='fast';const fast=chooseDiscard(g,0);g.botStyles[0]='defensive';const safe=chooseDiscard(g,0),obs=observe(g,0);
  assert.notEqual(fast,safe);assert.equal(Math.floor(safe/9),1);assert.equal(tileRisk(safe,obs),0);assert.ok(tileRisk(fast,obs)>0);
});
test('a previously discarded face is not a universal safe-tile rule',()=>{
  const g=fixture(),obs=observe(g,0);obs.opponents=[{que:2,won:false,melds:[],discards:[2],handCount:13}];assert.ok(tileRisk(2,obs)>0);obs.opponents[0].won=true;assert.equal(tileRisk(2,obs),0);
});
test('private opponent tiles and wall ordering cannot influence any profile decision',()=>{
  for(const respond of [null,0]){
    const a=fixture(respond==null?{}:{hand:[0,0,1,2,2,3,4,4,9,9,10,11,11],respond}),b=structuredClone(a);
    [b.players[1].hand[0],b.wall[0]]=[b.wall[0],b.players[1].hand[0]];b.wall.reverse();assertGame(b);assert.deepEqual(observe(a,0),observe(b,0));
    for(const style of styles){a.botStyles[0]=style;b.botStyles[0]=style;if(respond==null){assert.deepEqual(botDiscardAnalysis(a,0),botDiscardAnalysis(b,0));assert.deepEqual(chooseSelfAction(a,0),chooseSelfAction(b,0));}else assert.equal(chooseResponse(a,0),chooseResponse(b,0));}
  }
});
test('all profiles accept legal wins and cannot bypass a passed-hu lock',()=>{
  const hand=[0,1,2,3,4,5,6,7,8,9,9,9,12];
  for(const style of styles){const self=fixture({hand:[...hand,12]}),ron=fixture({hand,respond:12});self.botStyles[0]=style;ron.botStyles[0]=style;assert.equal(chooseSelfAction(self,0).type,'hu');assert.equal(chooseResponse(ron,0),'hu');ron.players[0].passedHuFan=5;assert.notEqual(chooseResponse(ron,0),'hu');}
});
test('fast callers open a useful hand while big-hand callers preserve five pairs',()=>{
  const g=fixture({hand:[0,0,1,2,2,3,4,4,9,9,10,11,11],respond:0});g.botStyles[0]='fast';assert.equal(chooseResponse(g,0),'pong');g.botStyles[0]='big';assert.equal(chooseResponse(g,0),'pass');
});
test('defensive play can decline an exposed added kong under public pressure',()=>{
  const g=fixture({hand:[0,3,4,5,9,10,11,12,12,14,15],melds:[{type:'pong',tile:0,from:2}],opponentMelds:[3,6,8].map(tile=>({type:'pong',tile,from:2}))});g.botStyles[0]='fast';assert.equal(chooseSelfAction(g,0)?.type,'addkong');g.botStyles[0]='defensive';assert.equal(chooseSelfAction(g,0),undefined);
});
test('every profile completes every variant with legal discards and physical conservation',()=>{
  for(const style of styles)for(const swap of [false,true])for(const joker of [false,true]){
    const g=createGame(17,null,{swap,joker,botStyles:Array(4).fill(style)});if(swap)exchangeTiles(g,g.players.map(p=>recommendSwap(p.hand,g.rules)));setAllQue(g,g.players.map(p=>recommendQue(p.hand,g.rules)));
    for(let step=0;step<1000&&g.phase!=='ended';step++){
      if(g.phase==='respond')resolveResponses(g,[0,1,2,3].map(s=>chooseResponse(g,s)));
      else{const a=chooseSelfAction(g,g.turn);if(a)selfAction(g,a.type,a.tile,g.turn,a.jokers??0);else{const t=chooseDiscard(g,g.turn);if(joker)assert.notEqual(t,18);discard(g,t,g.turn);}}
      g.events=[];assertGame(g);
    }assert.equal(g.phase,'ended',style+' '+swap+' '+joker);
  }
});
test('only a room host can configure bot seats and configuration clears all readiness',()=>{
  const r=new Room();r.init('BOTS2345','房主');r.join('真人');let id=0;const cmd=(s,action,extra={})=>r.command(s,{id:'b-'+(++id),action,...extra});
  cmd(0,'ready');cmd(1,'ready');const botStyles=['balanced','balanced','fast','big'];assert.throws(()=>cmd(1,'bot-config',{botStyles}),/房主/);assert.throws(()=>cmd(0,'bot-config',{botStyles:['balanced','fast','fast','big']}),/真人托管/);assert.throws(()=>cmd(0,'bot-config',{botStyles:['balanced','balanced','bad','big']}),/无效/);
  cmd(0,'bot-config',{botStyles});assert.deepEqual(r.view(0).botStyles,botStyles);assert.ok(r.d.members.filter(Boolean).every(m=>!m.ready));assert.deepEqual(new Room(r.serialize()).view(1).botStyles,botStyles);
  cmd(0,'ready');cmd(1,'ready');cmd(0,'start');assert.deepEqual(r.view(1).game.botStyles,botStyles);assert.throws(()=>cmd(0,'bot-config',{botStyles}),/锁定/);
  cmd(1,'auto',{value:true});assert.equal(r.d.game.botStyles[1],'balanced');
});
test('fixed multi-round matches lock personalities between rounds and old rooms migrate safely',()=>{
  const r=new Room();r.init('BOTS2345','房主');r.command(0,{id:'format',action:'match-config',rounds:4});r.command(0,{id:'ready',action:'ready'});r.command(0,{id:'start',action:'start'});finish(r.d.game);r.collectLogs();r.newDecision();assert.throws(()=>r.command(0,{id:'styles',action:'bot-config',botStyles:[...DEFAULT_BOT_STYLES]}),/锁定/);
  const old=r.serialize();delete old.botStyles;delete old.game.botStyles;assert.deepEqual(new Room(old).view(0).botStyles,Array(4).fill('balanced'));
});
test('bot pickers escape names and expose only the intended bot seats',()=>{
  const html=botPicker([...DEFAULT_BOT_STYLES],{names:['人','<script>','云','竹'],seats:[1,3]});assert.match(html,/&lt;script&gt;/);assert.equal((html.match(/data-bot-seat=/g)||[]).length,2);assert.doesNotMatch(html,/data-bot-seat="0"/);assert.match(botPicker([], {seats:[]}),/本桌均为真人/);
});
