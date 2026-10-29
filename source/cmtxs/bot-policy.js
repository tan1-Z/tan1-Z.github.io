// This module accepts observations and legal candidates only. It has no game,
// opponent-hand, wall-order, RNG, or network access.
export const BOT_STYLES=Object.freeze({
  balanced:Object.freeze({name:'均衡型',short:'均衡',description:'综合进张、成牌潜力和风险，打法稳定。'}),
  fast:Object.freeze({name:'快攻型',short:'快攻',description:'重视听牌速度，愿意碰牌加速，小胡也收。'}),
  big:Object.freeze({name:'做牌型',short:'做牌',description:'保留清一色、七对与碰碰胡机会，有条件地牺牲进张。'}),
  defensive:Object.freeze({name:'防守型',short:'防守',description:'公开威胁较高时，优先考虑风险较低的出牌。'}),
});
export const DEFAULT_BOT_STYLES=Object.freeze(['balanced','fast','big','defensive']);
export const isBotStyle=id=>typeof id==='string'&&Object.hasOwn(BOT_STYLES,id);
export const normalizeBotStyle=id=>isBotStyle(id)?id:'balanced';
export const normalizeBotStyles=(ids,fallback=DEFAULT_BOT_STYLES)=>Array.from({length:4},(_,s)=>normalizeBotStyle(Array.isArray(ids)&&isBotStyle(ids[s])?ids[s]:fallback[s]));
export const botStyleName=id=>BOT_STYLES[normalizeBotStyle(id)].name;
const clamp=(x,min=0,max=1)=>Math.max(min,Math.min(max,x));
const counts=h=>{const c=Array(27).fill(0);for(const t of h)c[t]++;return c;};
const isQue=(t,obs)=>Math.floor(t/9)===obs.que&&!(obs.rules?.joker&&t===18);
export function handPotential(obs,hand=obs.hand,melds=obs.melds){
  const h=hand.filter(t=>!isQue(t,obs)),j=obs.rules?.joker?h.filter(t=>t===18).length:0,natural=h.filter(t=>!(obs.rules?.joker&&t===18)),c=counts(natural);
  const pairs=c.reduce((n,v)=>n+Math.floor(v/2),0),triples=c.filter(v=>v>=3).length;
  let pure=0;
  for(let suit=0;suit<3;suit++){
    if(suit===obs.que||melds.some(m=>Math.floor(m.tile/9)!==suit))continue;
    const own=natural.filter(t=>Math.floor(t/9)===suit).length+melds.length*3,off=natural.length-natural.filter(t=>Math.floor(t/9)===suit).length;
    if(own+j>=8&&off<=4)pure=Math.max(pure,clamp((own+j-7)/6)*(1-off/10));
  }
  const seven=melds.length?0:clamp((pairs+Math.min(j,2)-3)/4);
  const triplet=clamp((triples+melds.length+c.filter(v=>v===2).length*.45+j*.25-1.5)/2.5);
  return {pure,seven,triplet,points:Math.max(pure*2,seven*2,triplet),pairs};
}
export function opponentThreat(opponent,remaining=55){
  if(opponent.won)return 0;
  return clamp(.12+(opponent.melds?.length||0)*.2+(1-clamp(remaining/55))*.32);
}
export const tableThreat=obs=>Math.max(0,...(obs.opponents||[]).map(q=>opponentThreat(q,obs.remaining)));
export function tileRisk(tile,obs){
  const suit=Math.floor(tile/9),rank=tile%9,seen=obs.seen||Array(27).fill(0),left=t=>clamp((4-(seen[t]||0))/4);
  let sequences=0;
  for(let start=Math.max(0,rank-2);start<=Math.min(6,rank);start++){
    const other=[start,start+1,start+2].filter(n=>n!==rank).map(n=>suit*9+n);
    sequences+=left(other[0])*left(other[1]);
  }
  const shape=.16+left(tile)*.22+sequences*.18;
  return (obs.opponents||[]).reduce((risk,q)=>{
    if(q.won||q.que===suit)return risk;
    let lean=1;const melds=q.melds||[];
    if(melds.length>=2&&melds.every(m=>Math.floor(m.tile/9)===Math.floor(melds[0].tile/9)))lean=Math.floor(melds[0].tile/9)===suit?1.35:.8;
    // Recent discards are a weak suit signal, never a furiten/safe-tile rule.
    if((q.discards||[]).slice(-4).filter(t=>Math.floor(t/9)===suit).length>=3)lean*=.9;
    return risk+opponentThreat(q,obs.remaining)*shape*lean;
  },0);
}
export function rankBotDiscards(obs,analysis,style='balanced'){
  style=normalizeBotStyle(style);if(!analysis.length)return [];
  const best=Math.min(...analysis.map(a=>a.shanten)),threat=tableThreat(obs),urgent=style==='defensive'&&threat>=.7&&best>=1;
  const rows=analysis.map(a=>{const after=[...obs.hand];after.splice(after.indexOf(a.tile),1);const potential=handPotential(obs,after),risk=tileRisk(a.tile,obs);return {...a,potential,risk};});
  const basePotential=Math.max(...rows.filter(a=>a.shanten===best).map(a=>a.potential.points));
  const candidates=rows.filter(a=>a.shanten===best||a.shanten===best+1&&(urgent||style==='big'&&best>=1&&obs.remaining>=24&&a.potential.points>=1.25&&a.potential.points-basePotential>=.45));
  const weights=style==='fast'?{s:140,u:1.65,p:0,r:.5}:style==='big'?{s:85,u:.8,p:52,r:5}:style==='defensive'?{s:urgent?30:120,u:.8,p:2,r:urgent?125:32}:{s:120,u:1.15,p:8,r:10};
  for(const a of candidates)a.preference=-a.shanten*weights.s+a.ukeire*weights.u+a.shape*.25+a.potential.points*weights.p-a.risk*weights.r;
  return candidates.sort((a,b)=>b.preference-a.preference||a.shanten-b.shanten||b.ukeire-a.ukeire||a.tile-b.tile);
}
export function chooseBotClaim(obs,candidates,current,ukeire,style='balanced'){
  style=normalizeBotStyle(style);const before=handPotential(obs),threat=tableThreat(obs);
  if(style==='big'&&!obs.melds.length&&before.seven>=.5&&obs.remaining>=20)return 'pass';
  const allowed=candidates.filter(a=>{
    if(a.jokers>1||a.shanten>current)return false;
    if(style==='big'&&before.points>=1.2&&a.potential.points+.3<before.points&&obs.remaining>=20)return false;
    if(style==='defensive'&&threat>=.7&&(a.shanten>0||a.risk>.35))return false;
    if(a.kind==='kong')return !a.jokers||style==='fast'&&a.shanten<current;
    if(a.shanten<current)return true;
    return style==='fast'&&current>0&&a.ukeire>=Math.max(8,ukeire*.9);
  });
  allowed.sort((a,b)=>a.shanten-b.shanten||(style==='big'?b.potential.points-a.potential.points:0)||b.ukeire-a.ukeire||a.risk-b.risk||a.jokers-b.jokers||a.action.localeCompare(b.action));
  return allowed[0]?.action||'pass';
}
export function chooseBotKong(obs,candidates,current,style='balanced'){
  style=normalizeBotStyle(style);const before=handPotential(obs),threat=tableThreat(obs);
  if(style==='big'&&!obs.melds.length&&before.seven>=.5&&obs.remaining>=20)return null;
  const allowed=candidates.filter(a=>!a.jokers&&a.shanten<=current&&(style!=='defensive'||threat<.7||a.shanten===0&&a.risk<.35));
  allowed.sort((a,b)=>a.shanten-b.shanten||(style==='defensive'?a.risk-b.risk:0)||a.tile-b.tile);
  return allowed[0]||null;
}
