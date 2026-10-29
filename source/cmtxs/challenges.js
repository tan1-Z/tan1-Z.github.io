import {normalizeRules,sorted,counts,meldTiles,isWinning,scoreHand,legalDiscards,selfActions,tileName,JOKER} from './engine.js?v=0.6.0';

// Offline, perfect-information puzzles: a discard advances exactly one fixed draw.
// They share legality and scoring with the game, without simulating other seats.
export const CHALLENGES=Object.freeze([
  {id:'last-pair',title:'最后一对',level:'入门',goal:'1 次摸牌内胡牌',note:'四组顺子已经齐了。最后两张，该留下哪一张作将？',hand:[0,1,2,3,4,5,9,10,11,12,13,14,6,8],draws:[6],que:2,solution:[{type:'discard',tile:8},{type:'hu'}],explain:'打出九万，保留七万；摸入七万后，以七万作将胡牌。'},
  {id:'pure-suit',title:'舍一门，成清一色',level:'入门',goal:'1 次摸牌内胡出清一色',note:'别让一张孤张，留住了一门花色。',hand:[0,1,2,3,4,5,6,7,8,0,0,4,4,9],draws:[4],que:2,tag:'清一色',solution:[{type:'discard',tile:9},{type:'hu'}],explain:'打出一筒，摸入五万。整副牌只剩万子，胡牌时得到清一色，并保留五万的根。'},
  {id:'seven-pairs',title:'七对的方向',level:'进阶',goal:'2 次摸牌内胡出七对',note:'六对已经在手，第一摸却不能成对。要为第二摸留住哪一张？',hand:[0,0,2,2,4,4,6,6,9,9,11,11,12,13],draws:[5,12],que:2,tag:'七对',solution:[{type:'discard',tile:13},{type:'discard',tile:5},{type:'hu'}],explain:'先打出五筒、保留四筒。第一摸六万无法成对，继续打出六万；第二摸四筒组成七个对子。'},
  {id:'clear-que',title:'先清缺，再用鸡',level:'进阶',goal:'1 次摸牌内清缺并胡牌',note:'幺鸡不受定缺限制，但其他缺门牌必须先打完。',joker:true,hand:[0,1,2,3,4,5,6,7,8,18,18,22,22,9],draws:[22],que:1,solution:[{type:'discard',tile:9},{type:'hu'}],explain:'先打出缺门的一筒，不能打幺鸡。摸入五条后，两张幺鸡可参与组成合法胡牌。'},
  {id:'bring-chicken-home',title:'把幺鸡换回来',level:'技巧',goal:'换回杠中的幺鸡后胡牌',note:'你刚摸入一万。已完成的杠中有一张幺鸡，可以换回；若它还在碰里，就不能这样换。',joker:true,hand:[0,1,2,3,9,10,11,12,12,12,4],drawn:0,draws:[8],que:2,required:'reclaim',melds:[{type:'kong',tile:0,from:1,tiles:[0,0,0,18]}],solution:[{type:'reclaim',tile:0},{type:'hu'}],explain:'用刚摸入的一万换回杠中的幺鸡。幺鸡回到暗手后可以补成五万对子；这次换回不额外摸牌。'},
  {id:'kong-flower',title:'杠后，那一朵花',level:'技巧',goal:'1 次摸牌内胡出杠上花',note:'暗杠后的补牌也消耗本题的一次摸牌机会。',hand:[0,0,0,0,3,4,5,9,10,11,12,12,12,7],draws:[7],que:2,tag:'杠上花',solution:[{type:'ankong',tile:0,jokers:0},{type:'hu'}],explain:'先暗杠一万，再补入八万组成将牌。补牌后立即胡牌，计杠上花。'},
]);
export const challengeById=id=>CHALLENGES.find(p=>p.id===id);
const copy=x=>structuredClone(x);
export function createChallenge(id=CHALLENGES[0].id){
  const p=challengeById(id);if(!p)throw Error('残局不存在');
  const s={id,hand:sorted(p.hand),melds:copy(p.melds||[]),que:p.que,rules:normalizeRules({swap:false,joker:!!p.joker}),drawn:p.drawn??p.hand.at(-1),drawIndex:0,afterKong:false,discards:[],moves:[],log:[],status:'playing',score:null};
  assertChallenge(s);return s;
}
function asGame(s){return {phase:s.status==='playing'?'discard':'ended',turn:0,rules:s.rules,drawn:s.drawn,wall:challengeById(s.id).draws.slice(s.drawIndex),players:[{hand:s.hand,melds:s.melds,que:s.que,won:false}]};}
export const challengeDiscards=s=>s.status==='playing'?legalDiscards(asGame(s),0):[];
export const challengeActions=s=>s.status==='playing'?selfActions(asGame(s),0):[];
export function challengeScore(s){return isWinning(s.hand,s.melds,s.que,s.rules)?scoreHand(s.hand,s.melds,{self:true,gang:s.afterKong,rules:s.rules,que:s.que}):null;}
export function challengeGoalMet(s){const p=challengeById(s.id),score=challengeScore(s);return !!score&&(!p.tag||score.tags.some(t=>t.name===p.tag))&&(!p.required||s.moves.some(m=>m.type===p.required));}
function remove(s,t,n,j=0){const used=[...Array(n-j).fill(t),...Array(j).fill(JOKER)];for(const x of used){const i=s.hand.indexOf(x);if(i<0)throw Error('手牌不足');s.hand.splice(i,1);}return used;}
function draw(s,kong=false){const p=challengeById(s.id);if(s.drawIndex>=p.draws.length){s.status='failed';s.log.push('摸牌次数用尽，本次挑战结束。');return;}const t=p.draws[s.drawIndex++];s.hand=sorted([...s.hand,t]);s.drawn=t;s.afterKong=kong;s.log.push(`${kong?'杠后补入':'摸入'}${tileName(t)}`);}
export function challengeMove(state,move){
  if(state.status!=='playing')throw Error('本次挑战已结束，请重新挑战');
  const s=copy(state),a={type:move.type,...(move.tile!=null?{tile:move.tile}:{}),...(['ankong','addkong'].includes(move.type)?{jokers:move.jokers??0}:{})};
  if(a.type==='discard'){
    if(!challengeDiscards(s).includes(a.tile))throw Error(s.rules.joker&&a.tile===JOKER?'幺鸡不能打出':'请先打完缺门牌');
    remove(s,a.tile,1);s.discards.push(a.tile);s.log.push('打出'+tileName(a.tile));s.moves.push(a);draw(s);
  }else{
    if(!challengeActions(s).some(x=>x.type===a.type&&(a.type==='hu'||x.tile===a.tile)&&(x.jokers??0)===(a.jokers??0)))throw Error('当前不能这样操作');
    if(a.type==='hu'){
      if(!challengeGoalMet(s))throw Error('当前胡牌尚未达到本题目标，可以继续调整');
      s.score=challengeScore(s);s.status='won';s.moves.push(a);s.log.push(`挑战成功 · ${s.score.fan} 番`);
    }else if(a.type==='reclaim'){
      const m=s.melds.find(m=>m.type!=='pong'&&m.tile===a.tile&&meldTiles(m).includes(JOKER));m.tiles=meldTiles(m);m.tiles[m.tiles.indexOf(JOKER)]=a.tile;
      remove(s,a.tile,1);s.hand=sorted([...s.hand,JOKER]);s.drawn=JOKER;s.moves.push(a);s.log.push('换回杠中的一张幺鸡');
    }else{
      const tiles=remove(s,a.tile,a.type==='ankong'?4:1,a.jokers);
      if(a.type==='ankong')s.melds.push({type:'ankong',tile:a.tile,from:0,tiles});
      else{const m=s.melds.find(m=>m.type==='pong'&&m.tile===a.tile);m.tiles=[...meldTiles(m),...tiles];m.type='addkong';}
      s.moves.push(a);s.log.push(`${a.type==='ankong'?'暗杠':'补杠'}${tileName(a.tile)}`);draw(s,true);
    }
  }
  assertChallenge(s);return s;
}
export function assertChallenge(s){
  const p=challengeById(s.id),tiles=[...s.hand,...s.discards,...s.melds.flatMap(meldTiles),...p.draws.slice(s.drawIndex)];
  if(tiles.some(t=>!Number.isInteger(t)||t<0||t>26)||counts(tiles).some(n=>n>4))throw Error('残局牌张无效');
  if(s.status!=='failed'&&s.hand.length!==14-s.melds.length*3)throw Error('残局手牌数量无效');
  return true;
}
export function restoreChallenge(saved){let s=createChallenge(saved?.id);if(!Array.isArray(saved?.moves)||saved.moves.length>80)throw Error('残局记录无效');for(const a of saved.moves)s=challengeMove(s,a);return s;}
