import {VERSION} from './engine.js?v=0.5.0';
const icons={
  table:'<rect x="3" y="3" width="18" height="18" rx="3"/><path d="M8 3v18M3 9h5M12 8h5m-5 4h5m-5 4h3"/>',
  review:'<path d="M4 4h12a3 3 0 0 1 3 3v13H7a3 3 0 0 1-3-3V4Z"/><path d="M4 16h15M8 8h7m-7 4h5M21 5v12"/>',
  train:'<path d="m12 3 3 6 6 3-6 3-3 6-3-6-6-3 6-3 3-6Z"/><path d="M12 8v8m-4-4h8"/>',
  users:'<circle cx="9" cy="8" r="3"/><path d="M3 20v-2a6 6 0 0 1 12 0v2m1-15a3 3 0 0 1 0 6m2 3a5 5 0 0 1 3 4v2"/>',
  rules:'<path d="M5 4h14v17l-7-4-7 4V4Z"/><path d="M9 8h6m-6 4h6"/>',
  plus:'<path d="M12 5v14M5 12h14"/>',
  help:'<circle cx="12" cy="12" r="9"/><path d="M9 9a3 3 0 0 1 6 0c0 2-3 2-3 4m0 3h.01"/>',
  sound:'<path d="m11 5-6 4H2v6h3l6 4V5Z"/><path d="M15 8a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14"/>',
  mute:'<path d="m11 5-6 4H2v6h3l6 4V5Z"/><path d="m16 9 6 6m-6 0 6-6"/>',
  arrow:'<path d="M4 12h16m-6-6 6 6-6 6"/>',
  back:'<path d="m14 6-6 6 6 6"/>',
  forward:'<path d="m10 6 6 6-6 6"/>',
  spark:'<path d="m12 3 2.6 6.4L21 12l-6.4 2.6L12 21l-2.6-6.4L3 12l6.4-2.6L12 3Z"/>',
  info:'<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10h.01"/>',
  leaf:'<path d="M19 3C6 3 2 8 6 14s13 1 13-11Z"/><path d="m5 20 10-11"/>',
  download:'<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',
  upload:'<path d="M12 16V4m-5 5 5-5 5 5M4 16v5h16v-5"/>',
  close:'<path d="m6 6 12 12M6 18 18 6"/>',
  pause:'<path d="M8 5v14M16 5v14"/>',
  play:'<path d="m8 4 12 8-12 8V4Z"/>',
  check:'<path d="m5 12 4 4L20 5"/>',
};
export const icon=(name)=>`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name]||icons.info}</svg>`;
export const PAGE_TITLES=Object.freeze({live:'人机练习',rooms:'好友开房',challenge:'残局挑战',review:'逐步复盘',train:'何切训练',rules:'规则手册'});
export const NAV_ITEMS=Object.freeze([
  {id:'live',icon:'table',label:PAGE_TITLES.live,href:'./'},
  {id:'rooms',icon:'users',label:PAGE_TITLES.rooms,href:'./rooms.html'},
  {id:'challenge',icon:'spark',label:PAGE_TITLES.challenge,href:'./#challenge'},
  {id:'review',icon:'review',label:PAGE_TITLES.review,href:'./#review'},
  {id:'train',icon:'train',label:PAGE_TITLES.train,href:'./#train'},
  {id:'rules',icon:'rules',label:PAGE_TITLES.rules,href:'./#rules'},
]);
export function pageFromHash(hash){const id=String(hash||'').replace(/^#/,'');return ['live','challenge','review','train','rules'].includes(id)?id:'live';}
export function renderSidebar(page='live',context='local'){
  const links=NAV_ITEMS.map(item=>`<a class="navbutton ${page===item.id?'active':''}" href="${item.href}" ${item.id==='rooms'?'':`data-page="${item.id}"`} ${page===item.id?'aria-current="page"':''}>${icon(item.icon)}<span>${item.label}</span></a>`).join('');
  return `<aside class="sidebar"><a class="brand" href="./" data-page="live" aria-label="听川首页"><div class="brandmark">听</div><div><div class="brandname">听川</div><div class="brandlatin">TINGCHUAN</div></div></a><div class="navlabel">川 麻 研 习 室</div><nav class="nav" aria-label="主导航">${links}</nav><div class="sidebar-note"><div class="line"></div>一张一弛，<br>慢慢听见牌的答案。</div><div class="sidebar-foot"><span><i class="statusdot"></i>${context==='online'?'好友房':'本地研习'}</span><span>v${VERSION}</span></div></aside>`;
}
export function renderHeader(page,controls=''){
  return `<header class="topbar"><div class="breadcrumb"><span>研习室</span><span>/</span><strong>${PAGE_TITLES[page]||PAGE_TITLES.live}</strong></div><a class="mobile-brand" href="./" data-page="live" aria-label="听川首页">听川<small>川麻研习室</small></a><div class="top-actions"><span class="rule-pill">${icon('leaf')}血战到底 · 自摸加番 · 五番满</span><span class="divider"></span>${controls}</div></header>`;
}
export function renderFooter(context='local'){
  return `<footer class="footer"><span class="wordmark">听川 · 以牌会意</span><span>听川 v${VERSION} · ${context==='online'?'房间保留 24 小时':'单机牌谱保存在本机'}</span></footer>`;
}
