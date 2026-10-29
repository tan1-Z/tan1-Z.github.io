import test from 'node:test';
import assert from 'node:assert/strict';
import {Room} from '../room.js';
import {newMatch,recordRound,matchView} from '../match.js';
import {createGame,finish} from '../../../source/cmtxs/engine.js';
import {reportText,renderMatch} from '../../../source/cmtxs/match-ui.js';
function setup(){let time=1000,seed=42,id=0;const options={now:()=>time,random:()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;},timeout:5,botDelay:1};let r=new Room(null,options);r.init('TEST2345','房主');r.join('好友');const cmd=(seat,action,extra={})=>r.command(seat,{id:'match-'+(++id),action,decision:r.d.decision,...extra});return {get r(){return r;},cmd,options,restart(){r=new Room(r.serialize(),options);},tick(){time+=10;r.tick();}};}

test('only the host can choose a valid format, which clears readiness',()=>{
  const {r,cmd}=setup();cmd(0,'ready');assert.throws(()=>cmd(1,'match-config',{rounds:4}),/房主/);assert.throws(()=>cmd(0,'match-config',{rounds:5}),/无效/);cmd(0,'match-config',{rounds:8});assert.equal(r.view(0).match.rounds,8);assert.equal(r.d.members[0].ready,false);
});
test('four real mixed rounds persist once, lock rules, finish, and archive on a fresh match',()=>{
  const f=setup();f.cmd(0,'match-config',{rounds:4});const savedIds=[];
  for(let round=1;round<=4;round++){
    f.cmd(0,'ready');f.cmd(1,'ready');f.cmd(0,'start');f.cmd(0,'auto',{value:true});f.cmd(1,'auto',{value:true});
    assert.throws(()=>f.cmd(0,'match-config',{rounds:8}),/锁定/);assert.throws(()=>f.cmd(0,'new-match',{rounds:8}),/完成当前/);
    for(let i=0;i<3000&&f.r.d.game.phase!=='ended';i++)f.tick();assert.equal(f.r.d.game.phase,'ended');
    const v=f.r.view(0);assert.equal(v.match.played,round);assert.equal(v.match.totals.reduce((a,b)=>a+b,0),0);savedIds.push(v.match.results.at(-1).id);
    if(round<4)assert.throws(()=>f.cmd(0,'configure',{rules:{swap:false,joker:true}}),/锁定/);
    f.restart();assert.equal(f.r.view(1).match.played,round);f.r.newDecision();assert.equal(f.r.view(1).match.played,round);
  }
  assert.equal(new Set(savedIds).size,4);assert.equal(f.r.view(0).match.completed,true);f.cmd(0,'ready');f.cmd(1,'ready');assert.throws(()=>f.cmd(0,'start'),/已结束/);
  const old=f.r.view(0).match;f.cmd(0,'new-match',{rounds:8});assert.equal(f.r.d.game,null);assert.equal(f.r.view(0).match.played,0);assert.deepEqual(f.r.view(0).match.totals,[0,0,0,0]);assert.deepEqual(f.r.view(0).matchArchive[0],old);assert.ok(f.r.d.members.filter(Boolean).every(m=>!m.ready));
});
test('repeated settlement and duplicate start commands cannot double count or advance a round',()=>{
  const r=new Room();r.init('TEST2345','房主');r.command(0,{id:'r',action:'ready'});r.command(0,{id:'s',action:'start'});const roundId=r.d.roundId;assert.equal(r.command(0,{id:'s',action:'start'}).duplicate,true);assert.equal(r.d.roundId,roundId);
  finish(r.d.game);r.collectLogs();r.newDecision();r.newDecision();assert.equal(r.view(0).match.played,1);
});
test('legacy rooms migrate to free play with one settled round and no hidden hand in reports',()=>{
  const r=new Room();r.init('TEST2345','旧房主');r.d.game=createGame(5);finish(r.d.game);const data=r.serialize();delete data.match;delete data.matchArchive;delete data.roundId;
  const restored=new Room(data),view=restored.view(0);assert.equal(view.match.rounds,0);assert.equal(view.match.played,1);assert.equal(new Room(restored.serialize()).view(0).match.played,1);assert.doesNotMatch(JSON.stringify(view.match),/"(hand|obs|wall|token|seed)"/);
});
test('tied totals share ranks and reports include actual fan, point income, and escaped names',()=>{
  const m=newMatch(4),g={phase:'ended',rules:{},ledger:[{from:2,to:0,amount:8,kind:'win'}],players:[0,1,2,3].map((seat)=>({name:seat===0?'<b>甲</b>':'玩家'+seat,score:[10,10,-10,-10][seat],won:seat===0,win:seat===0?{fan:3,from:2,self:false,tags:[]}:null}))};
  assert.equal(recordRound(m,g,1),true);assert.equal(recordRound(m,g,1),false);const v=matchView(m);assert.deepEqual(v.ranks,[1,1,3,3]);assert.equal(v.fires[2],1);assert.equal(v.biggest.points,8);assert.match(reportText(v,[],'TEST2345'),/阶段战报/);assert.match(renderMatch({match:v,members:[],seat:0},{full:true}),/&lt;b&gt;甲&lt;\/b&gt;/);
});
test('a trailing seat taking the cumulative lead gets a measured comeback',()=>{
  const m=newMatch(4),add=(id,scores)=>recordRound(m,{phase:'ended',rules:{},ledger:[],players:scores.map((score,seat)=>({name:'座'+seat,score,won:false}))},id);add(1,[10,-10,0,0]);add(2,[-25,25,0,0]);assert.deepEqual(matchView(m).comeback,{seat:1,round:2,gap:20});
});
