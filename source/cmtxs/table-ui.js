import {tileName,meldTiles,JOKER} from './engine.js?v=0.3.0';
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const positions=['south','east','north','west'];
const relativeNames=['你的','下家的','对家的','上家的'];
export function renderDiscardArena(game,viewer,drawTile){
  const max=Math.max(0,...game.players.map(p=>p.discards.length)),columns=max>60?12:max>32?10:8,rows=Math.max(3,Math.ceil(max/columns));
  // All four rotated grids fit inside the arena, including a long late-game river.
  const extent=2*Math.max(columns+(columns-1)*.08-1.9,1.9+.32+rows*1.4+(rows-1)*.08);
  const rivers=positions.map((position,relative)=>{
    const seat=(viewer+relative)%4,p=game.players[seat];
    return `<div class="seat-river river-${position}" data-seat="${seat}" role="group" aria-label="${relativeNames[relative]}弃牌，${esc(p.name)}，共${p.discards.length}张"><div class="discard-grid">${p.discards.map((t,index)=>{
      const latest=game.last&&!game.last.rob&&game.last.from===seat&&game.last.tile===t&&index===p.discards.length-1;
      return `<span class="river-card ${latest?'latest':''}" data-discard-index="${index+1}" title="${esc(p.name)} · 第${index+1}张 · ${tileName(t)}">${drawTile(t)}</span>`;
    }).join('')}</div></div>`;
  }).join('');
  const scores=positions.map((position,relative)=>{const seat=(viewer+relative)%4,p=game.players[seat],score=p.score||0;return `<span class="hub-score hub-${position} ${game.phase==='discard'&&game.turn===seat?'current':''}" title="${esc(p.name)}：${score}分">${score>0?'+':''}${score}</span>`;}).join('');
  return `<div class="discard-arena" style="--river-cols:${columns};--river-extent:${extent.toFixed(3)}" role="group" aria-label="四方弃牌区">${rivers}<div class="discard-hub">${scores}<div class="hub-wall" aria-label="牌墙剩余${game.remaining}张"><strong>${game.remaining}</strong><small>余牌</small></div></div></div>`;
}
export function renderMeldBadges(melds,drawTile){
  if(!melds.length)return '';
  return `<div class="table-meld-badges">${melds.map(m=>{const j=meldTiles(m).filter(t=>t===JOKER).length,label=`${m.type==='pong'?'碰':'杠'}${tileName(m.tile)}${j?`，用${j}张幺鸡`:''}`;return `<span class="table-meld-badge" title="${label}" aria-label="${label}">${drawTile(m.tile)}<small>${m.type==='pong'?'碰':'杠'}${j?j+'鸡':''}</small></span>`;}).join('')}</div>`;
}
