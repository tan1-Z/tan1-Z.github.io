import {EMOTES,EMOTE_COOLDOWN_MS,EMOTE_DURATION_MS,getEmote,EmoteFeed} from './emotes.js?v=0.6.0';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const smile='<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M8 14s1 3 4 3 4-3 4-3M8 9h.01M16 9h.01"/></svg>';
export function emotePanel({open=false,muted=false,enabled=true,cooldown=0}={}){
  return `<button type="button" class="emote-open" data-emote-action="toggle" aria-expanded="${open}" aria-controls="table-emote-panel" aria-label="打开表情面板">${smile}<span>表情${cooldown?' · '+Math.ceil(cooldown/1000)+'s':''}</span></button><button type="button" class="emote-mute ${muted?'is-muted':''}" data-emote-action="mute" aria-pressed="${muted}" title="${muted?'恢复显示同桌表情':'隐藏同桌表情'}" aria-label="${muted?'恢复显示同桌表情':'隐藏同桌表情'}">${muted?'显':'隐'}</button>${open?`<section class="table-emote-panel" id="table-emote-panel" role="dialog" aria-label="发送牌桌表情"><header><div><strong>以表情，会牌友。</strong><small>点击发送 · 每 3 秒一次</small></div><button type="button" class="emote-close" data-emote-action="close" aria-label="关闭表情面板">×</button></header><div class="emote-grid">${EMOTES.map(e=>`<button type="button" class="emote-choice" data-emote-id="${e.id}" aria-label="发送表情：${e.label}" ${!enabled||cooldown?'disabled':''}><img src="${e.src}" alt="" width="96" height="96" loading="lazy" decoding="async"><span>${e.label}</span><small>${e.group}</small></button>`).join('')}</div><p class="emote-panel-note">${!enabled?'连接恢复后可发送。':muted?'已隐藏表情；发送后同桌仍可看到。':'表情在座位旁短暂显示，不影响行牌。'}</p></section>`:''}`;
}
export function createEmoteController({send,notice=()=>{},now=Date.now}={}){
  const feed=new EmoteFeed({now});let root=null,viewer=0,names=[],enabled=true,open=false,lastSent=-Infinity,timer=null,focusId=null;
  try{feed.setMuted(localStorage.getItem('tingchuan-emotes-muted-v1')==='true');}catch{}
  const cooldown=()=>Math.max(0,EMOTE_COOLDOWN_MS-(now()-lastSent));
  function refresh({focus=false}={}){
    clearTimeout(timer);if(!root?.isConnected)return;
    const host=root.querySelector('[data-emote-controls]'),layer=root.querySelector('[data-emote-layer]');
    if(host){const focused=host.contains(document.activeElement)?document.activeElement?.getAttribute('data-emote-id')||document.activeElement?.getAttribute('data-emote-action'):null;host.innerHTML=emotePanel({open,muted:feed.muted,enabled,cooldown:cooldown()});if(focus||focused){const key=focus?focusId:focused;host.querySelector(key&&getEmote(key)?`[data-emote-id="${key}"]`:key?`[data-emote-action="${key}"]`:'.emote-choice:not(:disabled)')?.focus({preventScroll:true});}}
    const panel=host?.querySelector('.table-emote-panel');
    if(panel){const table=root.getBoundingClientRect();panel.style.maxWidth=Math.max(160,table.width-24)+'px';panel.style.maxHeight=Math.max(140,Math.min(480,host.getBoundingClientRect().top-table.top-22))+'px';const box=panel.getBoundingClientRect(),shift=box.left<table.left+10?table.left+10-box.left:box.right>table.right-10?table.right-10-box.right:0;panel.style.transform=`translateX(${shift}px)`;if(focus&&open)panel.scrollIntoView({block:'nearest',behavior:'instant'});}
    const active=feed.current();
    if(layer){const rect=root.getBoundingClientRect(),size=rect.width<480?84:110;layer.innerHTML=active.map(e=>{const anchor=root.querySelector(`[data-emote-seat="${e.seat}"]`);if(!anchor)return '';const box=anchor.getBoundingClientRect(),x=Math.max(7,Math.min(rect.width-size-7,box.left-rect.left+box.width/2-size/2)),y=e.seat===viewer?box.top-rect.top-size-7:box.bottom-rect.top+5;const image=getEmote(e.sticker);return `<figure class="table-emote ${now()-e.at<250?'just-sent':''}" data-emote-key="${e.key}" data-emote-seat-display="${e.seat}" style="left:${x}px;top:${Math.max(7,Math.min(rect.height-size-7,y))}px;width:${size}px"><img src="${image.src}" alt="${esc(names[e.seat]||'牌友')}：${image.label}" width="${size}" height="${size}"></figure>`;}).join('');}
    const waits=[...active.map(e=>e.until-now()),...(cooldown()?[Math.min(1000,cooldown())]:[])].filter(n=>n>0);if(waits.length)timer=setTimeout(refresh,Math.min(...waits)+5);
  }
  function mount(table,context={}){
    root=table;viewer=context.viewer??0;names=context.names||[];enabled=context.enabled!==false;
    if(!root){open=false;feed.clear();clearTimeout(timer);return;}
    refresh({focus:open&&!!focusId&&document.activeElement===document.body});
  }
  function receive(event,offset=0){
    if(!root||!feed.receive(event,offset))return false;refresh();
    const live=root.querySelector('[data-emote-announcer]');if(live)live.textContent=`${names[event.seat]||'牌友'}发送了表情：${getEmote(event.sticker).label}`;
    return true;
  }
  function handle(event){
    if(!root)return false;
    const control=event.target.closest('[data-emote-action],[data-emote-id]');
    if(!control||!root.contains(control)){if(open&&!event.target.closest('[data-emote-controls]')){open=false;refresh();}return false;}
    const id=control.getAttribute('data-emote-id'),action=control.getAttribute('data-emote-action');
    if(id){
      if(!enabled){notice('正在恢复连接，请稍候。');return true;}
      if(cooldown()){notice('表情冷却中，请稍候。');return true;}
      if(send(id)!==false){lastSent=now();open=false;focusId='toggle';refresh({focus:true});}return true;
    }
    if(action==='toggle'){open=!open;focusId=null;refresh({focus:open});}
    if(action==='close'){open=false;focusId='toggle';refresh({focus:true});}
    if(action==='mute'){feed.setMuted(!feed.muted);try{localStorage.setItem('tingchuan-emotes-muted-v1',String(feed.muted));}catch{}focusId='mute';refresh({focus:true});notice(feed.muted?'已隐藏牌桌表情，仍可正常发送。':'已恢复显示牌桌表情。');}
    return true;
  }
  function keydown(event){if(open&&event.key==='Escape'){event.preventDefault();open=false;focusId='toggle';refresh({focus:true});return true;}return !!event.target.closest('[data-emote-controls]');}
  function clear(){open=false;feed.clear();refresh();}
  window.addEventListener('resize',()=>refresh());
  document.addEventListener('focusin',e=>{if(root?.contains(e.target)&&e.target.closest('[data-emote-controls]'))focusId=e.target.getAttribute('data-emote-id')||e.target.getAttribute('data-emote-action');});
  window.addEventListener('storage',e=>{if(e.key==='tingchuan-emotes-muted-v1'){feed.setMuted(e.newValue==='true');refresh();}});
  return {mount,receive,handle,keydown,clear,controls:()=>'<div class="table-emotes" data-emote-controls></div>',layer:()=>'<div class="table-emote-layer" data-emote-layer></div><span class="emote-announcer" data-emote-announcer role="status" aria-live="polite"></span>',localEvent:(sticker,seat=0)=>({type:'emote',id:crypto.randomUUID(),sticker,seat,at:now(),expiresAt:now()+EMOTE_DURATION_MS})};
}
