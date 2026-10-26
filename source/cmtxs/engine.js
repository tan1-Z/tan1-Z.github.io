export const VERSION = '0.2.0';
export const isCompatibleVersion = version => ['0.1.0','0.1.1',VERSION].includes(version);
export const RULES = Object.freeze({ version:'xuezhan-1', cap:5, zimoFan:1, swap:true, dingque:true, huazhu:true, dajiao:true, refund:true, transfer:false, passHu:true });
export const SUITS = ['万','筒','条'];
export const tileName = t => `${t % 9 + 1}${SUITS[Math.floor(t / 9)]}`;
export const sorted = hand => [...hand].sort((a,b) => a-b);
export const counts = hand => { const a=Array(27).fill(0); for(const t of hand)a[t]++; return a; };
const clone = x => structuredClone(x);
const suitCache = new Map();

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
export function shanten(hand, open=0) {
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
export function isWinning(hand,melds=[],que=-1) {
  if(hand.some(t=>Math.floor(t/9)===que)||melds.some(m=>Math.floor(m.tile/9)===que))return false;
  if(hand.length!==14-melds.length*3)return false;
  const c=counts(hand);
  if(c.some(x=>x>4))return false;
  if(!melds.length&&c.every(x=>x%2===0))return true;
  for(let i=0;i<27;i++)if(c[i]>=2){c[i]-=2;const ok=complete(c,4-melds.length);c[i]+=2;if(ok)return true;}
  return false;
}
export function scoreHand(hand,melds=[],{self=false,gang=false,rob=false,last=false}={}) {
  const c=counts(hand), all=[...hand,...melds.flatMap(m=>Array(m.type==='pong'?3:4).fill(m.tile))];
  const allC=counts(all), tags=[]; let fan=0;
  const add=(name,n)=>{tags.push({name,fan:n});fan+=n;};
  const seven=!melds.length&&c.every(x=>x%2===0);
  if(seven)add('七对',2);
  else if(c.every(x=>[0,2,3].includes(x))&&c.filter(x=>x===2).length===1)add('碰碰胡',1);
  if(new Set(all.map(t=>Math.floor(t/9))).size===1)add('清一色',2);
  if(melds.length===4)add('金钩钓',1);
  const roots=allC.filter(x=>x===4).length;
  if(roots)add(seven?'根（龙七对）':'根',roots);
  if(gang)add('杠上花',1);
  if(rob)add('抢杠胡',1);
  if(last)add('海底',1);
  if(self)add('自摸',RULES.zimoFan);
  if(!tags.length)tags.push({name:'平胡',fan:0});
  return {fan:Math.min(fan,RULES.cap),rawFan:fan,points:2**Math.min(fan,RULES.cap),tags};
}
export function observe(g,seat) {
  const p=g.players[seat], seen=counts(p.hand);
  for(let s=0;s<4;s++){
    const q=g.players[s];
    for(const t of q.discards)seen[t]++;
    for(const m of q.melds)seen[m.tile]+=m.type==='pong'?3:4;
    // Winning hands are public; ron tile is already present in the discard pool.
    if(s!==seat&&q.won)for(const t of q.hand)seen[t]++;
  }
  return {hand:[...p.hand],melds:clone(p.melds),que:p.que,seen,remaining:g.wall.length};
}
export function analyze(obs) {
  const {hand,melds,que,seen}=obs;
  const forced=hand.filter(t=>Math.floor(t/9)===que);
  const legal=[...new Set(forced.length?forced:hand)];
  return legal.map(tile=>{
    const after=[...hand];after.splice(after.indexOf(tile),1);
    const effective=after.filter(t=>Math.floor(t/9)!==que);
    const s=shanten(effective,melds.length), improves=[];
    for(let t=0;t<27;t++){
      if(Math.floor(t/9)===que)continue;
      const left=Math.max(0,4-seen[t]);if(!left)continue;
      if(shanten([...effective,t],melds.length)<s)improves.push({tile:t,count:left});
    }
    const ukeire=improves.reduce((n,v)=>n+v.count,0);
    const c=counts(effective), shape=c.reduce((v,n,i)=>v+(n>=2?1.4:0)+(i%9<8&&n&&c[i+1]?.8:0),0);
    return {tile,shanten:s,ukeire,improves,shape,forced:forced.length>0};
  }).sort((a,b)=>a.shanten-b.shanten||b.ukeire-a.ukeire||b.shape-a.shape||a.tile-b.tile);
}
export function recommendQue(hand){
  return [0,1,2].map(s=>({s,n:hand.filter(t=>Math.floor(t/9)===s).length})).sort((a,b)=>a.n-b.n||a.s-b.s)[0].s;
}
export function recommendSwap(hand){
  const s=recommendQue(hand), candidates=hand.map((t,i)=>({t,i})).filter(x=>Math.floor(x.t/9)===s);
  if(candidates.length>=3)return candidates.slice(0,3).map(x=>x.i);
  return hand.map((t,i)=>({t,i})).filter(x=>Math.floor(x.t/9)===[0,1,2].find(s=>hand.filter(t=>Math.floor(t/9)===s).length>=3)).slice(0,3).map(x=>x.i);
}
export function randomSeed(){const a=new Uint32Array(1);globalThis.crypto.getRandomValues(a);return a[0];}
function rng(seed){let a=seed>>>0;return()=>{a+=0x6D2B79F5;let t=a;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};}
export function createGame(seed=randomSeed(),randomSource=null) {
  const rand=randomSource||rng(seed), wall=Array.from({length:108},(_,i)=>Math.floor(i/4));
  for(let i=107;i>0;i--){const j=Math.floor(rand()*(i+1));[wall[i],wall[j]]=[wall[j],wall[i]];}
  const players=['你','竹隐','云岫','松风'].map((name,seat)=>({name,seat,hand:[],melds:[],discards:[],que:-1,won:false,score:0,passedHuFan:null}));
  for(let n=0;n<13;n++)for(const p of players)p.hand.push(wall.pop());
  players[0].hand.push(wall.pop());for(const p of players)p.hand=sorted(p.hand);
  return {version:VERSION,ruleVersion:RULES.version,seed,players,wall,turn:0,phase:'swap',swapDirection:1+Math.floor(rand()*3),last:null,drawn:null,afterKong:false,ledger:[],events:[],message:'选择同一花色的三张牌，换一份新的可能。',winners:[],decision:null};
}
export function snapshot(g){const s=clone(g);s.events=[];return s;}
function log(g,type,text,extra={}){g.events.push({id:g.events.length,type,text,...extra,state:snapshot(g)});}
export function swapTiles(g,indices) {
  return exchangeTiles(g,g.players.map((p,s)=>s===0?indices:recommendSwap(p.hand)));
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
  return setAllQue(g,g.players.map((p,s)=>s===0?suit:recommendQue(p.hand)));
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
export function legalDiscards(g,s=g.turn){const p=g.players[s],forced=p.hand.filter(t=>Math.floor(t/9)===p.que);return [...new Set(forced.length?forced:p.hand)];}
function clearQue(p){return !p.hand.some(t=>Math.floor(t/9)===p.que);}
export function selfActions(g,s=g.turn){
  if(g.phase!=='discard'||g.turn!==s||g.players[s].won)return [];
  const p=g.players[s],out=[];if(g.drawn!==null&&isWinning(p.hand,p.melds,p.que))out.push({type:'hu'});
  if(clearQue(p)&&g.wall.length){const c=counts(p.hand);for(let t=0;t<27;t++){
    if(c[t]===4)out.push({type:'ankong',tile:t});
    if(c[t]&&p.melds.some(m=>m.type==='pong'&&m.tile===t))out.push({type:'addkong',tile:t});
  }}return out;
}
function claimOptions(g,s,t,from,rob=false){
  const p=g.players[s];if(p.won||s===from||!clearQue(p)||Math.floor(t/9)===p.que)return [];
  const a=[];
  if(isWinning([...p.hand,t],p.melds,p.que)){
    const fan=ronScore(g,s,t,rob).fan;
    if(p.passedHuFan==null||fan>p.passedHuFan)a.push('hu');
  }
  if(!rob){const n=p.hand.filter(x=>x===t).length;if(n>=2)a.push('pong');if(n===3&&g.wall.length)a.push('kong');}
  return a;
}
function ronScore(g,s,t,rob=false){
  const p=g.players[s];
  return scoreHand([...p.hand,t],p.melds,{rob,last:!g.wall.length});
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
    if(!p||p.won||!last||last.from===s||!isWinning([...p.hand,last.tile],p.melds,p.que))continue;
    let tookWin=false;
    for(let j=i+1;j<events.length&&events[j].type==='win';j++)if(events[j].seat===s)tookWin=true;
    if(!tookWin)fan=Math.max(fan??-1,ronScore(state,s,last.tile,last.rob).fan);
  }
  return fan;
}
export function restoreGame(g){
  for(let s=0;s<g.players.length;s++){
    const p=g.players[s];
    if(!Object.hasOwn(p,'passedHuFan'))p.passedHuFan=legacyPassedHuFan(g,s);
  }
  return refreshDecision(g);
}
export function discard(g,t,s=g.turn){
  if(g.phase!=='discard'||g.turn!==s||!legalDiscards(g,s).includes(t))throw Error('这张牌当前不能打出，请先打完定缺花色');
  const p=g.players[s];let review=null;
  if((g.reviewSeats||[0]).includes(s)){const obs=observe(g,s),analysis=analyze(obs);review={obs,analysis,chosen:t};}
  p.hand.splice(p.hand.indexOf(t),1);p.discards.push(t);g.last={tile:t,from:s,rob:false};g.phase='respond';g.drawn=null;g.afterKong=false;
  refreshDecision(g);const actions=g.decision?.actions||[];
  g.message=actions.length?`${p.name}打出${tileName(t)}，你可以响应。`:`${p.name}打出${tileName(t)}`;
  log(g,'discard',`${p.name}打出${tileName(t)}`,{seat:s,tile:t,review});
}
function win(g,s,t,from,{self=false,rob=false}={}){
  const p=g.players[s],hand=self?p.hand:[...p.hand,t];
  const scored=self?scoreHand(hand,p.melds,{self:true,gang:g.afterKong,last:!g.wall.length}):ronScore(g,s,t,rob);
  const payers=self?active(g).filter(x=>x.seat!==s).map(x=>x.seat):[from];
  for(const f of payers)pay(g,f,s,scored.points,self?'自摸':'点炮');
  p.won=true;p.win={tile:t,from,self,...scored};g.winners.push(s);
  log(g,'win',`${p.name}${self?'自摸':'胡牌'} · ${scored.fan}番 · +${scored.points*payers.length}`,{seat:s});
}
function takeClaim(g,s,type,t,from){
  const p=g.players[s],n=type==='pong'?2:3;
  for(let i=0;i<n;i++)p.hand.splice(p.hand.indexOf(t),1);
  g.players[from].discards.pop();p.melds.push({type:type==='pong'?'pong':'kong',tile:t,from});
  g.turn=s;g.last=null;g.decision=null;g.afterKong=false;g.drawn=null;
  log(g,type,`${p.name}${type==='pong'?'碰':'杠'}${tileName(t)}`);
  if(type==='kong'){pay(g,from,s,2,'直杠','kong');draw(g,s,true);}else{g.phase='discard';g.message=s===0?'碰牌成功，请出牌。':`${p.name}碰牌后出牌`;}
}
function aiClaim(g,s,t,from,rob){
  const options=claimOptions(g,s,t,from,rob),p=g.players[s];if(options.includes('hu'))return 'hu';
  if(options.includes('kong'))return 'kong';
  if(options.includes('pong')){const hand=[...p.hand];for(let i=0;i<2;i++)hand.splice(hand.indexOf(t),1);
    const current=shanten(p.hand,p.melds.length),after=Math.min(...[...new Set(hand)].map(x=>{const h=[...hand];h.splice(h.indexOf(x),1);return shanten(h,p.melds.length+1);}));
    if(after<current)return 'pong';
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
    if(rob){const p=g.players[from];p.hand.splice(p.hand.indexOf(t),1);p.discards.push(t);}
    for(const {s} of wins)win(g,s,t,from,{rob});
    if(active(g).length<=1){finish(g);return;}
    draw(g,nextSeat(g,from));return;
  }
  if(rob){completeAddKong(g,from,t);return;}
  const claim=choices.find(c=>['pong','kong'].includes(c.action));
  if(claim){takeClaim(g,claim.s,claim.action,t,from);return;}
  draw(g,nextSeat(g,from));
}
function completeAddKong(g,s,t){
  const p=g.players[s];p.hand.splice(p.hand.indexOf(t),1);p.melds.find(m=>m.tile===t&&m.type==='pong').type='addkong';
  for(const q of active(g))if(q.seat!==s)pay(g,q.seat,s,1,'补杠','kong');
  log(g,'kong',`${p.name}补杠${tileName(t)}`);draw(g,s,true);
}
export function selfAction(g,type,t,s=g.turn){
  if(!selfActions(g,s).some(a=>a.type===type&&(type==='hu'||a.tile===t)))throw Error('当前操作不合法');
  const p=g.players[s];
  if(type==='hu'){win(g,s,g.drawn,s,{self:true});if(active(g).length<=1)finish(g);else draw(g,nextSeat(g,s));return;}
  if(type==='ankong'){
    for(let i=0;i<4;i++)p.hand.splice(p.hand.indexOf(t),1);p.melds.push({type:'ankong',tile:t,from:s});
    for(const q of active(g))if(q.seat!==s)pay(g,q.seat,s,2,'暗杠','kong');
    log(g,'kong',`${p.name}暗杠${tileName(t)}`);draw(g,s,true);return;
  }
  g.last={tile:t,from:s,rob:true};g.phase='respond';refreshDecision(g);
  g.message=`${p.name}尝试补杠${tileName(t)}`;
  log(g,'addkong',g.message);
}
export function aiStep(g){
  if(g.phase==='respond'){refreshDecision(g);if(g.decision)return false;resolveResponse(g);return true;}
  if(g.phase!=='discard'||g.turn===0)return false;
  const acts=selfActions(g),hu=acts.find(a=>a.type==='hu');if(hu){selfAction(g,'hu');return true;}
  const kong=acts.find(a=>a.type==='ankong'||a.type==='addkong');if(kong){selfAction(g,kong.type,kong.tile);return true;}
  const analysis=analyze(observe(g,g.turn));discard(g,analysis[0].tile);return true;
}
export function waits(p){
  if(!clearQue(p))return [];
  const allC=counts([...p.hand,...p.melds.flatMap(m=>Array(m.type==='pong'?3:4).fill(m.tile))]);
  return Array.from({length:27},(_,t)=>t).filter(t=>allC[t]<4&&isWinning([...p.hand,t],p.melds,p.que));
}
export function finish(g){
  if(g.phase==='ended')return;
  if(!g.wall.length&&active(g).length>1){
    const left=active(g), pigs=left.filter(p=>!clearQue(p)),nonpigs=g.players.filter(p=>clearQue(p));
    for(const p of pigs)for(const q of nonpigs)pay(g,p.seat,q.seat,2**RULES.cap,'查花猪','settlement');
    const ready=left.filter(p=>!pigs.includes(p)&&waits(p).length),unready=left.filter(p=>!pigs.includes(p)&&!waits(p).length);
    for(const q of ready){const max=Math.max(...waits(q).map(t=>scoreHand([...q.hand,t],q.melds).points));for(const p of unready)pay(g,p.seat,q.seat,max,'查大叫','settlement');}
    const refundSeats=[...pigs,...unready].map(p=>p.seat);
    for(const tax of [...g.ledger])if(tax.kind==='kong'&&!tax.refunded&&refundSeats.includes(tax.to)){
      tax.refunded=true;pay(g,tax.to,tax.from,tax.amount,'未听牌退税','refund');
    }
  }
  g.phase='ended';g.decision=null;g.message=g.wall.length?'三家已胡，本局结束。':'牌墙已空，完成查叫结算。';log(g,'end',g.message);
}
export function assertGame(g){
  const c=counts([...g.wall,...g.players.flatMap(p=>[...p.hand,...p.discards,...p.melds.flatMap(m=>Array(m.type==='pong'?3:4).fill(m.tile))])]);
  if(c.some(x=>x!==4))throw Error('牌张不守恒：'+c.join(','));
  if(g.players.reduce((n,p)=>n+p.score,0)!==0)throw Error('分数不守恒');
  return true;
}
