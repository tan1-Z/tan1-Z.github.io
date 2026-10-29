import test from 'node:test';
import assert from 'node:assert/strict';
import {CHALLENGES,createChallenge,challengeMove,challengeDiscards,challengeActions,restoreChallenge,assertChallenge,challengeGoalMet} from '../../../source/cmtxs/challenges.js';
import {pageFromHash,NAV_ITEMS,renderSidebar} from '../../../source/cmtxs/ui-shell.js';

test('all six physical puzzles have a legal, target-satisfying reference solution',()=>{
  for(const p of CHALLENGES){let s=createChallenge(p.id);for(const a of p.solution){s=challengeMove(s,a);assertChallenge(s);}assert.equal(s.status,'won',p.id);assert.ok(s.drawIndex<=p.draws.length);if(p.tag)assert.ok(s.score.tags.some(t=>t.name===p.tag));}
});
test('a wrong pair choice fails without silently granting a win or an extra draw',()=>{
  let s=challengeMove(createChallenge('last-pair'),{type:'discard',tile:6});assert.equal(s.drawIndex,1);assert.ok(!challengeActions(s).some(a=>a.type==='hu'));
  s=challengeMove(s,{type:'discard',tile:0});assert.equal(s.status,'failed');assert.equal(s.drawIndex,1);assert.throws(()=>challengeMove(s,{type:'hu'}),/已结束/);
});
test('missing-suit enforcement and joker discard prohibition remain atomic',()=>{
  const s=createChallenge('clear-que'),before=structuredClone(s);assert.deepEqual(challengeDiscards(s),[9]);assert.throws(()=>challengeMove(s,{type:'discard',tile:18}),/不能打出/);assert.throws(()=>challengeMove(s,{type:'discard',tile:0}),/缺门/);assert.deepEqual(s,before);
});
test('a joker reclaim uses no draw and remains available only once on that draw',()=>{
  let s=createChallenge('bring-chicken-home');s=challengeMove(s,{type:'reclaim',tile:0});assert.equal(s.drawIndex,0);assert.equal(s.melds[0].tiles.filter(t=>t===18).length,0);assert.ok(!challengeActions(s).some(a=>a.type==='reclaim'));assert.throws(()=>challengeMove(s,{type:'reclaim',tile:0}));
});
test('a legal win cannot bypass the special objective',()=>{
  const s=challengeMove(createChallenge('kong-flower'),{type:'discard',tile:0}),before=structuredClone(s);assert.ok(challengeActions(s).some(a=>a.type==='hu'));assert.equal(challengeGoalMet(s),false);assert.throws(()=>challengeMove(s,{type:'hu'}),/未达到/);assert.deepEqual(s,before);
});
test('restoring a puzzle replays checked actions and does not trust a saved success flag',()=>{
  const s=restoreChallenge({id:'last-pair',status:'won',moves:[]});assert.equal(s.status,'playing');assert.equal(restoreChallenge({id:'last-pair',moves:CHALLENGES[0].solution}).status,'won');assert.throws(()=>restoreChallenge({id:'not-a-puzzle',moves:[]}));assert.throws(()=>restoreChallenge({id:'last-pair',moves:[{type:'discard',tile:99}]}));
});
test('the new navigation route exists identically in both page contexts',()=>{
  assert.equal(pageFromHash('#challenge'),'challenge');assert.equal(NAV_ITEMS.length,6);for(const context of ['local','online'])assert.match(renderSidebar('challenge',context),/href="\.\/#challenge"[^>]+aria-current="page"/);
});
