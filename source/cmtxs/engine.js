import {JOKER,canComplete,bestAssignment} from './joker.js?v=0.3.0';
export {JOKER};
export const VERSION = '0.3.0';
export const isCompatibleVersion = version => ['0.1.0','0.1.1','0.2.0',VERSION].includes(version);
export const RULES = Object.freeze({ version:'xuezhan-1', cap:5, zimoFan:1, swap:true, joker:false, dingque:true, huazhu:true, dajiao:true, refund:true, transfer:false, passHu:true });
export const normalizeRules = options => ({...RULES,swap:options?.swap!==false,joker:options?.joker===true});
export const modeName = options => `${options?.joker?'幺鸡':'血战'}${options?.swap===false?'定缺 · 不换三张':'换三张 · 定缺'}`;
export const meldTiles = m => m.tiles?[...m.tiles]:Array(m.type==='pong'?3:4).fill(m.tile);
export const isQueTile = (t,que,rules=RULES) => Math.floor(t/9)===que&&!(rules.joker&&t===JOKER);
export const clearQue = (p,rules=RULES) => !p.hand.some(t=>isQueTile(t,p.que,rules));
export const SUITS = ['万','筒','条'];
export const tileName = t => `${t % 9 + 1}${SUITS[Math.floor(t / 9)]}`;
export const sorted = hand => [...hand].sort((a,b) => a-b);
export const counts = hand => { const a=Array(27).fill(0); for(const t of hand)a[t]++; return a; };
const clone = x => structuredClone(x);
const suitCache = new Map();
const jokerScoreCache = new Map();

// Enumerate every relevant decomposition of one suit. No opponents' hands are used.
function suitShapes(c) {
  const key=c.join('');
  if(suitCache.has(key))return suitCache.get(key);
  const found=new Set();
  function walk(m,t,p) {
    let i=c.findIndex(x=>x>0);
    if(i<0){found.add(`${m},${t},${p}`);return;}
    c[i]--;walk(m,t,p);c[i]++;
    if(c[i]>=3){c[i]-=3;walk(m+1,t,p);c[i]+=3;}
    if(i<7&&c[i+1]&&c[i+2]){c[i]--;c[i+1]--;c[i+2]--;walk(m+1,t,p);c[i]++;c[i+1]++;c[i+2]++;}
    if(c[i]>=2){c[i]-=2;walk(m,t+1,p);if(!p)walk(m,t,1);c[i]+=2;}
    for(const d of [1,2])if(i+d<9&&c[i+d]){c[i]--;c[i+d]--;walk(m,t+1,p);c[i]++;c[i+d]++;}
  }
  walk(0,0,0);
  const list=[...found].map(v=>v.split(',').map(Number));
  // For a fixed number of melds and pair, only the largest taatsu count matters.
  const best=new Map();
  for(const [m,t,p] of list){const k=m+','+p;if(!best.has(k)||best.get(k)[1]<t)best.set(k,[m,t,p]);}
  const result=[...best.values()];
  if(suitCache.size>60000)suitCache.clear();
  suitCache.set(key,result);return result;
}
export function shanten(hand, open=0, rules=RULES) {
  if(rules.joker&&hand.includes(JOKER)){
    const j=hand.filter(t=>t===JOKER).length,natural=hand.filter(t=>t!==JOKER);
    const floor=hand.length+3*open===14?-1:0;
    const value=Math.max(floor,shanten(natural,open)-j);
    return value===-1&&!canComplete(hand,open,true)?0:value;
  }
  const c=counts(hand); const ss=[0,1,2].map(s=>suitShapes(c.slice(s*9,s*9+9)));
  let best=8;
  for(const a of ss[0])for(const b of ss[1])for(const d of ss[2]){
    const m=open+a[0]+b[0]+d[0], p=a[2]+b[2]+d[2];
    if(p>1||m>4)continue;
    const t=Math.min(a[1]+b[1]+d[1],4-m);
    best=Math.min(best,8-m*2-t-p);
  }
  if(!open)best=Math.min(best,6-c.reduce((n,v)=>n+Math.floor(v/2),0));
  return best;
}
function complete(c, remaining) {
  if(!remaining)return c.every(x=>!x);
  const i=c.findIndex(x=>x>0);if(i<0)return false;
  if(c[i]>=3){c[i]-=3;const ok=complete(c,remaining-1);c[i]+=3;if(ok)return true;}
  if(i%9<=6&&c[i+1]&&c[i+2]){c[i]--;c[i+1]--;c[i+2]--;const ok=complete(c,remaining-1);c[i]++;c[i+1]++;c[i+2]++;if(ok)return true;}
  return false;
}
export function isWinning(hand,melds=[],que=-1,rules=RULES) {
  if(hand.some(t=>isQueTile(t,que,rules))||melds.some(m=>Math.floor(m.tile/9)===que))return false;
  if(hand.length!==14-melds.length*3)return false;
  if(rules.joker)return canComplete(hand,melds.length,true);
  const c=counts(hand);
  if(c.some(x=>x>4))return false;
  if(!melds.length&&c.every(x=>x%2===0))return true;
  for(let i=0;i<27;i++)if(c[i]>=2){c[i]-=2;const ok=complete(c,4-melds.length);c[i]+=2;if(ok)return true;}
  return false;
}
export function scoreHand(hand,melds=[],{self=false,gang=false,rob=false,last=false,rules=RULES,que=-1,noChicken=true}={}) {
  if(rules.joker&&hand.includes(JOKER)){
    const key=JSON.stringify([sorted(hand),melds,que,self,gang,rob,last]);
    if(jokerScoreCache.has(key))return clone(jokerScoreCache.get(key));
    const scored=bestAssignment(hand,melds.length,que,h=>scoreHand(h,melds,{self,gang,rob,last,rules:{...rules,joker:false},que}));
    if(jokerScoreCache.size>256)jokerScoreCache.clear();jokerScoreCache.set(key,scored);return clone(scored);
  }
  const c=counts(hand), all=[...hand,...melds.flatMap(m=>Array(m.type==='pong'?3:4).fill(m.tile))];
  const allC=counts(all), tags=[]; let fan=0;
  const add=(name,n)=>{tags.push({name,fan:n});fan+=n;};
  const seven=!melds.length&&c.every(x=>x%2===0);
  if(seven)add('七对',2);
  else if(c.filter(x=>x%3===2).length===1&&c.every(x=>x%3!==1))add('碰碰胡',1);
  if(new Set(all.map(t=>Math.floor(t/9))).size===1)add('清一色',2);
  if(melds.length===4)add('金钩钓',1);
  const roots=allC.reduce((n,v)=>n+Math.floor(v/4),0);
  if(roots)add(seven?'根（龙七对）':'根',roots);
  if(gang)add('杠上花',1);
  if(rob)add('抢杠胡',1);
  if(last)add('海底',1);
  if(self)add('自摸',RULES.zimoFan);
  if(rules.joker&&noChicken&&!hand.includes(JOKER)&&!melds.some(m=>meldTiles(m).includes(JOKER)))add('无鸡',1);
  if(!tags.length)tags.push({name:'平胡',fan:0});
  return {fan:Math.min(fan,RULES.cap),rawFan:fan,points:2**Math.min(fan,RULES.cap),tags};
}
export function observe(g,seat) {
  const p=g.players[seat], seen=counts(p.hand);
  for(let s=0;s<4;s++){
    const q=g.players[s];
    for(const t of q.discards)seen[t]++;
    for(const m of q.melds)for(const t of meldTiles(m))seen[t]++;
    // Winning hands are public; ron tile is already present in the discard pool.
    if(s!==seat&&q.won)for(const t of q.hand)seen[t]++;
  }
  return {hand:[...p.hand],melds:clone(p.melds),que:p.que,seen,remaining:g.wall.length,rules:normalizeRules(g.rules)};
}
export function analyze(obs) {
  const {hand,melds,que,seen}=obs,rules=normalizeRules(obs.rules);
  const forced=hand.filter(t=>isQueTile(t,que,rules));
  const legal=[...new Set(forced.length?forced:hand.filter(t=>!(rules.joker&&t===JOKER)))];
  return legal.map(tile=>{
    const after=[...hand];after.splice(after.indexOf(tile),1);
    const effective=after.filter(t=>!isQueTile(t,que,rules));
    const s=shanten(effective,melds.length,rules), improves=[];
    for(let t=0;t<27;t++){
      if(isQueTile(t,que,rules))continue;
      const left=Math.max(0,4-seen[t]);if(!left)continue;
      if(shanten([...effective,t],melds.length,rules)<s)improves.push({tile:t,count:left});
    }
    const ukeire=improves.reduce((n,v)=>n+v.count,0);
    const c=counts(effective), shape=c.reduce((v,n,i)=>v+(n>=2?1.4:0)+(i%9<8&&n&&c[i+1]?.8:0),0);
    return {tile,shanten:s,ukeire,improves,shape,forced:forced.length>0};
  }).sort((a,b)=>a.shanten-b.shanten||b.ukeire-a.ukeire||b.shape-a.shape||a.tile-b.tile);
}
export function recommendQue(hand,rules=RULES){
  return [0,1,2].map(s=>({s,n:hand.filter(t=>isQueTile(t,s,rules)).length})).sort((a,b)=>a.n-b.n||a.s-b.s)[0].s;
}
export function recommendSwap(hand,rules=RULES){
  const s=recommendQue(hand,rules), candidates=hand.map((t,i)=>({t,i})).filter(x=>Math.floor(x.t/9)===s).sort((a,b)=>(rules.joker&&a.t===JOKER)-(rules.joker&&b.t===JOKER)||a.t-b.t);
  if(candidates.length>=3)return candidates.slice(0,3).map(x=>x.i);
  return hand.map((t,i)=>({t,i})).filter(x=>Math.floor(x.t/9)===[0,1,2].find(s=>hand.filter(t=>Math.floor(t/9)===s).length>=3)).slice(0,3).map(x=>x.i);
}
export function randomSeed(){const a=new Uint32Array(1);globalThis.crypto.getRandomValues(a);return a[0];}
function rng(seed){let a=seed>>>0;return()=>{a+=0x6D2B79F5;let t=a;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};}
export function createGame(seed=randomSeed(),randomSource=null,options=RULES) {
  const rules=normalizeRules(options);
  const rand=randomSource||rng(seed), wall=Array.from({length:108},(_,i)=>Math.floor(i/4));
  for(let i=107;i>0;i--){const j=Math.floor(rand()*(i+1));[wall[i],wall[j]]=[wall[j],wall[i]];}
  const players=['你','竹隐','云岫','松风'].map((name,seat)=>({name,seat,hand:[],melds:[],discards:[],que:-1,won:false,score:0,passedHuFan:null}));
  for(let n=0;n<13;n++)for(const p of players)p.hand.push(wall.pop());
  players[0].hand.push(wall.pop());for(const p of players)p.hand=sorted(p.hand);
  return {version:VERSION,ruleVersion:RULES.version,rules,seed,players,wall,turn:0,phase:rules.swap?'swap':'que',swapDirection:1+Math.floor(rand()*3),last:null,drawn:null,afterKong:false,ledger:[],events:[],message:rules.swap?'选择同一花色的三张牌，换一份新的可能。':'本局不换三张，请直接定缺。',winners:[],decision:null};
}
export function snapshot(g){const s=clone(g);s.events=[];return s;}
function log(g,type,text,extra={}){g.events.push({id:g.events.length,type,text,...extra,state:snapshot(g)});}
export function swapTiles(g,indices) {
  return exchangeTiles(g,g.players.map((p,s)=>s===0?indices:recommendSwap(p.hand,g.rules)));
}
// Validate all four submissions before changing any hand.
export function exchangeTiles(g,chosen) {
  if(g.phase!=='swap')throw Error('当前不能换牌');
  if(!Array.isArray(chosen)||chosen.length!==4)throw Error('需要四家的换牌');
  chosen.forEach((indices,s)=>{
    if(!Array.isArray(indices)||indices.length!==3||new Set(indices).size!==3||indices.some(i=>!Number.isInteger(i)||i<0||i>=g.players[s].hand.length))throw Error('请选择三张牌');
    const tiles=indices.map(i=>g.players[s].hand[i]);if(new Set(tiles.map(t=>Math.floor(t/9))).size!==1)throw Error('换出的三张牌必须属于同一花色');
  });
  const packets=chosen.map((arr,s)=>arr.map(i=>g.players[s].hand[i]));
  for(let s=0;s<4;s++){for(const i of [...chosen[s]].sort((a,b)=>b-a))g.players[s].hand.splice(i,1);}
  for(let s=0;s<4;s++)g.players[(s+g.swapDirection)%4].hand.push(...packets[s]);
  for(const p of g.players)p.hand=sorted(p.hand);
  g.phase='que';g.message='换牌完成，请选择本局的定缺花色。';log(g,'swap','四家完成换三张');
}
export function setQue(g,suit){
  return setAllQue(g,g.players.map((p,s)=>s===0?suit:recommendQue(p.hand,g.rules)));
}
export function setAllQue(g,suits){
  if(g.phase!=='que'||!Array.isArray(suits)||suits.length!==4||suits.some(s=>![0,1,2].includes(s)))throw Error('当前不能定缺');
  g.players.forEach((p,s)=>p.que=suits[s]);
  g.phase='discard';g.drawn=g.players[0].hand.at(-1);g.message='定缺完成，庄家出牌。';log(g,'que','四家定缺完成，开始行牌');
}
function pay(g,from,to,amount,reason,kind='win'){
  g.players[from].score-=amount;g.players[to].score+=amount;
  g.ledger.push({from,to,amount,reason,kind,refunded:false});
}
function active(g){return g.players.filter(p=>!p.won);}
function nextSeat(g,from){for(let n=1;n<=4;n++){const s=(from+n)%4;if(!g.players[s].won)return s;}return from;}
function draw(g,s,after=false){
  if(active(g).length<=1||!g.wall.length){finish(g);return;}
  g.turn=s;const t=g.wall.pop();g.players[s].hand.push(t);g.players[s].hand=sorted(g.players[s].hand);
  // Only this player's own draw (including a kong replacement) clears the lock.
  g.players[s].passedHuFan=null;
  g.drawn=t;g.afterKong=after;g.phase='discard';g.last=null;g.message=s===0?'轮到你出牌。':`${g.players[s].name}正在思考…`;
  log(g,'draw',`${g.players[s].name}摸牌`,{seat:s});
}
export function legalDiscards(g,s=g.turn){const p=g.players[s],rules=normalizeRules(g.rules),forced=p.hand.filter(t=>isQueTile(t,p.que,rules));return [...new Set(forced.length?forced:p.hand.filter(t=>!(rules.joker&&t===JOKER)))];}
export function selfActions(g,s=g.turn){
  if(g.phase!=='discard'||g.turn!==s||g.players[s].won)return [];
  const p=g.players[s],rules=normalizeRules(g.rules),out=[];
  if(g.drawn!==null&&isWinning(p.hand,p.melds,p.que,rules))out.push({type:'hu'});
  if(!clearQue(p,rules))return out;
  if(rules.joker&&g.drawn!=null&&g.drawn!==JOKER&&p.hand.includes(g.drawn)&&p.melds.some(m=>m.type!=='pong'&&m.tile===g.drawn&&meldTiles(m).includes(JOKER)))out.push({type:'reclaim',tile:g.drawn});
  if(g.wall.length)for(let t=0;t<27;t++){
    if(isQueTile(t,p.que,rules)||rules.joker&&t===JOKER)continue;
    for(const j of meldUses(p,t,4,rules))out.push({type:'ankong',tile:t,jokers:j});
    if(p.melds.some(m=>m.type==='pong'&&m.tile===t))for(const j of meldUses(p,t,1,rules))out.push({type:'addkong',tile:t,jokers:j});
  }
  return out;
}
function meldUses(p,t,n,rules){
  const natural=p.hand.filter(x=>x===t).length,jokers=rules.joker?p.hand.filter(x=>x===JOKER).length:0,out=[];
  for(let j=0;j<=Math.min(jokers,n);j++)if(n-j<=natural)out.push(j);
  return out;
}
export const claimType = action => String(action).split(':')[0];
export const claimJokers = action => Number(String(action).split(':')[1]||0);
const claimCode = (type,j) => j?`${type}:${j}`:type;
function removeMeldTiles(hand,t,n,j){
  const used=[...Array(n-j).fill(t),...Array(j).fill(JOKER)];
  for(const x of used)hand.splice(hand.indexOf(x),1);
  return used;
}
const ronTile = (g,t) => g.last?.rob&&g.last.tile===t?(g.last.physical??t):t;
function claimOptions(g,s,t,from,rob=false){
  const p=g.players[s],rules=normalizeRules(g.rules);
  if(p.won||s===from||!clearQue(p,rules)||isQueTile(t,p.que,rules))return [];
  const a=[];
  if(isWinning([...p.hand,t],p.melds,p.que,rules)){
    const fan=ronScore(g,s,t,rob).fan;
    if(p.passedHuFan==null||fan>p.passedHuFan)a.push('hu');
  }
  if(!rob&&!(rules.joker&&t===JOKER)){
    for(const j of meldUses(p,t,2,rules)){
      const rest=[...p.hand];removeMeldTiles(rest,t,2,j);
      if(rest.some(x=>!(rules.joker&&x===JOKER)))a.push(claimCode('pong',j));
    }
    if(g.wall.length)for(const j of meldUses(p,t,3,rules))a.push(claimCode('kong',j));
  }
  return a;
}
function ronScore(g,s,t,rob=false){
  const p=g.players[s];
  return scoreHand([...p.hand,t],p.melds,{rob,last:!g.wall.length,rules:normalizeRules(g.rules),que:p.que,noChicken:ronTile(g,t)!==JOKER});
}
export function responseActions(g,s=0){
  if(g.phase!=='respond'||!g.last)return [];
  return claimOptions(g,s,g.last.tile,g.last.from,g.last.rob);
}
export function refreshDecision(g){
  const actions=responseActions(g);
  g.decision=actions.length?{...g.last,actions}:null;
  return g;
}
// v0.1.0 did not store refusals explicitly. Its event snapshots let us recover
// resolved winning opportunities since the last own draw without resetting play.
function legacyPassedHuFan(g,s){
  let fan=null;
  const events=g.events||[];
  for(let i=0;i<events.length;i++){
    const e=events[i];
    if(e.type==='draw'&&e.seat===s){fan=null;continue;}
    if(e.type==='pass-hu'&&e.seat===s){fan=Math.max(fan??-1,e.fan);continue;}
    if(!['discard','addkong'].includes(e.type)||i+1>=events.length)continue;
    const state=e.state,p=state?.players?.[s],last=state?.last;
    if(!p||p.won||!last||last.from===s||!isWinning([...p.hand,last.tile],p.melds,p.que,normalizeRules(state.rules)))continue;
    let tookWin=false;
    for(let j=i+1;j<events.length&&events[j].type==='win';j++)if(events[j].seat===s)tookWin=true;
    if(!tookWin)fan=Math.max(fan??-1,ronScore(state,s,last.tile,last.rob).fan);
  }
  return fan;
}
export function restoreGame(g){
  g.rules=normalizeRules(g.rules);
  for(let s=0;s<g.players.length;s++){
    const p=g.players[s];
    if(!Object.hasOwn(p,'passedHuFan'))p.passedHuFan=legacyPassedHuFan(g,s);
  }
  return refreshDecision(g);
}
export function discard(g,t,s=g.turn){
  if(g.phase!=='discard'||g.turn!==s||!legalDiscards(g,s).includes(t))throw Error(g.rules?.joker&&t===JOKER?'幺鸡是万能牌，不能打出':'这张牌当前不能打出，请先打完定缺花色');
  const p=g.players[s];let review=null;
  if((g.reviewSeats||[0]).includes(s)){const obs=observe(g,s),analysis=analyze(obs);review={obs,analysis,chosen:t};}
  p.hand.splice(p.hand.indexOf(t),1);p.discards.push(t);g.last={tile:t,from:s,rob:false};g.phase='respond';g.drawn=null;g.afterKong=false;
  refreshDecision(g);const actions=g.decision?.actions||[];
  g.message=actions.length?`${p.name}打出${tileName(t)}，你可以响应。`:`${p.name}打出${tileName(t)}`;
  log(g,'discard',`${p.name}打出${tileName(t)}`,{seat:s,tile:t,review});
}
function win(g,s,t,from,{self=false,rob=false}={}){
  const p=g.players[s],hand=self?p.hand:[...p.hand,t];
  const scored=self?scoreHand(hand,p.melds,{self:true,gang:g.afterKong,last:!g.wall.length,rules:normalizeRules(g.rules),que:p.que}):ronScore(g,s,t,rob);
  const payers=self?active(g).filter(x=>x.seat!==s).map(x=>x.seat):[from];
  for(const f of payers)pay(g,f,s,scored.points,self?'自摸':'点炮');
  p.won=true;p.win={tile:t,from,self,...scored};g.winners.push(s);
  log(g,'win',`${p.name}${self?'自摸':'胡牌'} · ${scored.fan}番 · +${scored.points*payers.length}`,{seat:s});
}
function takeClaim(g,s,action,t,from){
  const type=claimType(action),j=claimJokers(action),p=g.players[s],n=type==='pong'?2:3;
  const tiles=[t,...removeMeldTiles(p.hand,t,n,j)];
  g.players[from].discards.pop();p.melds.push({type:type==='pong'?'pong':'kong',tile:t,from,tiles});
  g.turn=s;g.last=null;g.decision=null;g.afterKong=false;g.drawn=null;
  log(g,type,`${p.name}${type==='pong'?'碰':'杠'}${tileName(t)}${j?`（用${j}张幺鸡）`:''}`);
  if(type==='kong'){pay(g,from,s,2,'直杠','kong');draw(g,s,true);}else{g.phase='discard';g.message=s===0?'碰牌成功，请出牌。':`${p.name}碰牌后出牌`;}
}
function aiClaim(g,s,t,from,rob){
  const options=claimOptions(g,s,t,from,rob),p=g.players[s];if(options.includes('hu'))return 'hu';
  if(options.includes('kong'))return 'kong';
  const rules=normalizeRules(g.rules),current=shanten(p.hand,p.melds.length,rules);
  for(const action of options.filter(a=>claimType(a)==='pong')){
    const hand=[...p.hand];removeMeldTiles(hand,t,2,claimJokers(action));
    const after=Math.min(...[...new Set(hand)].filter(x=>!(rules.joker&&x===JOKER)).map(x=>{const h=[...hand];h.splice(h.indexOf(x),1);return shanten(h,p.melds.length+1,rules);}));
    if(after<current)return action;
  }return 'pass';
}
export function resolveResponse(g,human='pass'){
  return resolveResponses(g,g.players.map((p,s)=>s===0?human:chooseResponse(g,s)));
}
export function chooseResponse(g,s){
  if(g.phase!=='respond'||!g.last)return 'pass';
  return aiClaim(g,s,g.last.tile,g.last.from,g.last.rob);
}
export function resolveResponses(g,responses){
  if(g.phase!=='respond')throw Error('当前没有待响应的牌');
  const {tile:t,from,rob}=g.last;
  const choices=[];
  for(let n=1;n<4;n++){const s=(from+n)%4;if(g.players[s].won)continue;const allowed=claimOptions(g,s,t,from,rob);
    const action=responses[s];
    if(action!=='pass'&&!allowed.includes(action)){
      if(action==='hu'&&g.players[s].passedHuFan!=null)throw Error(`过手胡限制：接炮须高于${g.players[s].passedHuFan}番，或等待自己摸牌。`);
      throw Error('当前不可进行此操作');
    }
    choices.push({s,action,passedFan:allowed.includes('hu')&&action!=='hu'?ronScore(g,s,t,rob).fan:null});
  }
  // Validate every action first: a rejected command must not create a refusal.
  for(const c of choices)if(c.passedFan!==null){
    const p=g.players[c.s];p.passedHuFan=Math.max(p.passedHuFan??-1,c.passedFan);
  }
  refreshDecision(g);
  for(const c of choices)if(c.passedFan!==null)log(g,'pass-hu',`${g.players[c.s].name}过胡 · ${c.passedFan}番`,{seat:c.s,fan:c.passedFan,tile:t});
  const wins=choices.filter(c=>c.action==='hu');g.decision=null;
  if(wins.length){
    if(rob){const p=g.players[from],physical=ronTile(g,t);p.hand.splice(p.hand.indexOf(physical),1);p.discards.push(physical);}
    for(const {s} of wins)win(g,s,t,from,{rob});
    if(active(g).length<=1){finish(g);return;}
    draw(g,nextSeat(g,from));return;
  }
  if(rob){completeAddKong(g,from,t);return;}
  const claim=choices.find(c=>['pong','kong'].includes(claimType(c.action)));
  if(claim){takeClaim(g,claim.s,claim.action,t,from);return;}
  draw(g,nextSeat(g,from));
}
function completeAddKong(g,s,t){
  const p=g.players[s],physical=ronTile(g,t),m=p.melds.find(m=>m.tile===t&&m.type==='pong');
  p.hand.splice(p.hand.indexOf(physical),1);m.tiles=[...meldTiles(m),physical];m.type='addkong';
  for(const q of active(g))if(q.seat!==s)pay(g,q.seat,s,1,'补杠','kong');
  log(g,'kong',`${p.name}补杠${tileName(t)}`);draw(g,s,true);
}
export function selfAction(g,type,t,s=g.turn,jokers=0){
  if(!selfActions(g,s).some(a=>a.type===type&&(type==='hu'||a.tile===t)&&(a.jokers??0)===jokers))throw Error('当前操作不合法');
  const p=g.players[s];
  if(type==='hu'){win(g,s,g.drawn,s,{self:true});if(active(g).length<=1)finish(g);else draw(g,nextSeat(g,s));return;}
  if(type==='reclaim'){
    const m=p.melds.find(m=>m.type!=='pong'&&m.tile===t&&meldTiles(m).includes(JOKER));m.tiles=meldTiles(m);m.tiles[m.tiles.indexOf(JOKER)]=t;
    p.hand.splice(p.hand.indexOf(t),1);p.hand.push(JOKER);p.hand=sorted(p.hand);g.drawn=JOKER;
    log(g,'reclaim',`${p.name}用${tileName(t)}换回杠中的一张幺鸡`);g.message='幺鸡已换回手牌，请继续操作。';return;
  }
  if(type==='ankong'){
    const tiles=removeMeldTiles(p.hand,t,4,jokers);p.melds.push({type:'ankong',tile:t,from:s,tiles});
    for(const q of active(g))if(q.seat!==s)pay(g,q.seat,s,2,'暗杠','kong');
    log(g,'kong',`${p.name}暗杠${tileName(t)}${jokers?`（用${jokers}张幺鸡）`:''}`);draw(g,s,true);return;
  }
  g.last={tile:t,physical:jokers?JOKER:t,from:s,rob:true};g.phase='respond';refreshDecision(g);
  g.message=`${p.name}尝试补杠${tileName(t)}`;
  log(g,'addkong',g.message);
}
export function aiStep(g){
  if(g.phase==='respond'){refreshDecision(g);if(g.decision)return false;resolveResponse(g);return true;}
  if(g.phase!=='discard'||g.turn===0)return false;
  const a=chooseSelfAction(g);if(a){selfAction(g,a.type,a.tile,g.turn,a.jokers??0);return true;}
  const analysis=analyze(observe(g,g.turn));discard(g,analysis[0].tile);return true;
}
export function chooseSelfAction(g,s=g.turn){
  const acts=selfActions(g,s);
  return acts.find(a=>a.type==='hu')||acts.find(a=>a.type==='reclaim')||acts.find(a=>!a.jokers&&['ankong','addkong'].includes(a.type));
}
export function waits(p,rules=RULES){
  if(!clearQue(p,rules))return [];
  const allC=counts([...p.hand,...p.melds.flatMap(meldTiles)]);
  return Array.from({length:27},(_,t)=>t).filter(t=>allC[t]<4&&isWinning([...p.hand,t],p.melds,p.que,rules));
}
export function finish(g){
  if(g.phase==='ended')return;
  if(!g.wall.length&&active(g).length>1){
    const rules=normalizeRules(g.rules),left=active(g), pigs=left.filter(p=>!clearQue(p,rules)),nonpigs=g.players.filter(p=>clearQue(p,rules));
    for(const p of pigs)for(const q of nonpigs)pay(g,p.seat,q.seat,2**RULES.cap,'查花猪','settlement');
    const ready=left.filter(p=>!pigs.includes(p)&&waits(p,rules).length),unready=left.filter(p=>!pigs.includes(p)&&!waits(p,rules).length);
    for(const q of ready){const max=Math.max(...waits(q,rules).map(t=>scoreHand([...q.hand,t],q.melds,{rules,que:q.que}).points));for(const p of unready)pay(g,p.seat,q.seat,max,'查大叫','settlement');}
    const refundSeats=[...pigs,...unready].map(p=>p.seat);
    for(const tax of [...g.ledger])if(tax.kind==='kong'&&!tax.refunded&&refundSeats.includes(tax.to)){
      tax.refunded=true;pay(g,tax.to,tax.from,tax.amount,'未听牌退税','refund');
    }
  }
  g.phase='ended';g.decision=null;g.message=g.wall.length?'三家已胡，本局结束。':'牌墙已空，完成查叫结算。';log(g,'end',g.message);
}
export function assertGame(g){
  const c=counts([...g.wall,...g.players.flatMap(p=>[...p.hand,...p.discards,...p.melds.flatMap(meldTiles)])]);
  if(c.some(x=>x!==4))throw Error('牌张不守恒：'+c.join(','));
  if(g.players.reduce((n,p)=>n+p.score,0)!==0)throw Error('分数不守恒');
  return true;
}
