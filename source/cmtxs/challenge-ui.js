import {CHALLENGES,challengeById,createChallenge,restoreChallenge,challengeMove,challengeDiscards,challengeActions,challengeScore,challengeGoalMet} from './challenges.js?v=0.4.0';
import {tileName,SUITS,meldTiles,isQueTile} from './engine.js?v=0.3.0';
import {tileFace} from './tile-faces.js?v=0.3.4';
import {selfLabel,meldCaption} from './variants-ui.js?v=0.3.0';
const KEY='tingchuan-challenges-v1';
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const button=(text,action,disabled=false,cls='')=>`<button type="button" class="button ${cls}" data-challenge="${action}" ${disabled?'disabled':''}>${text}</button>`;
export function challengeController({render,notice}){
  let saved={};try{saved=JSON.parse(localStorage.getItem(KEY))||{};}catch{}
  let state;try{state=restoreChallenge(saved.attempt);}catch{state=createChallenge();}
  state.assisted=!!saved.assisted||!!saved.hint;
  let selected=null,hint=!!saved.hint,bests={};
  for(const p of CHALLENGES){const b=saved.bests?.[p.id];if(b&&Number.isInteger(b.draws)&&b.draws>=0&&b.draws<=p.draws.length)bests[p.id]={draws:b.draws,assisted:!!b.assisted};}
  function persist(){try{localStorage.setItem(KEY,JSON.stringify({attempt:{id:state.id,moves:state.moves},hint,assisted:!!state.assisted,bests}));}catch{notice('本机存储不可用，本次仍可继续挑战。');}}
  function select(id,updateURL=true){if(!challengeById(id))return;state=createChallenge(id);selected=null;hint=false;persist();if(updateURL){const u=new URL(location.href);u.searchParams.set('challenge',id);u.hash='challenge';history.replaceState(null,'',u);} }
  function sync(){const id=new URL(location.href).searchParams.get('challenge');if(challengeById(id)&&id!==state.id)select(id,false);}
  sync();
  const face=(t,action=null,chosen=false)=>`<${action?'button type="button"':'span'} class="tile ${chosen?'selected':''} ${state.rules.joker&&t===18?'joker':''} ${isQueTile(t,state.que,state.rules)?'forced':''}" ${action?`data-challenge="${action}" aria-label="${tileName(t)}${chosen?'，已选择':''}" aria-pressed="${chosen}"`:''} title="${tileName(t)}">${tileFace(t)}${state.rules.joker&&t===18?'<b class="joker-mark">百搭</b>':''}</${action?'button':'span'}>`;
  function page(){
    const s=state,p=challengeById(s.id),legal=challengeDiscards(s),score=challengeScore(s),actions=challengeActions(s),goal=challengeGoalMet(s),record=bests[s.id];
    return `<section class="intro challenge-intro"><div><div class="eyebrow">ENDGAME CHALLENGE · ONE MORE GOOD CHOICE</div><h1>这几张，能打出什么？</h1><p>固定摸牌、明确目标。把一手牌想透，再邀请朋友来试。</p></div><div class="intro-actions">${button('分享本题','share')}</div></section><div class="challenge-layout"><aside class="panel challenge-list"><div class="panel-title">残局小册<span class="tinybadge">${Object.keys(bests).filter(id=>challengeById(id)).length} / ${CHALLENGES.length} 已通关</span></div>${CHALLENGES.map((q,i)=>`<button type="button" class="challenge-card ${q.id===s.id?'active':''}" data-challenge="open-${q.id}" ${q.id===s.id?'aria-current="true"':''}><span class="challenge-number">0${i+1}</span><span><strong>${q.title}</strong><small>${q.level} · ${q.joker?'幺鸡':'普通血战'}${bests[q.id]?' · 已通关':''}</small></span><span aria-hidden="true">${bests[q.id]?'✓':'↗'}</span></button>`).join('')}</aside><div class="challenge-main"><section class="board-wrap challenge-board"><div class="board-bar"><div class="board-bar-title"><strong>${p.title}</strong><span class="badge">${p.level}</span></div><span class="challenge-draw-count">摸牌 ${s.drawIndex} / ${p.draws.length}</span></div><div class="challenge-felt"><div class="challenge-goal"><span>本题目标</span><h2>${p.goal}</h2><p>${p.note}</p></div><div class="challenge-meta"><span>${p.joker?'幺鸡万能牌':'普通血战'} · 定缺${SUITS[s.que]}</span><span>固定摸牌序列</span></div><div class="challenge-draws" aria-label="固定摸牌序列">${p.draws.map((t,i)=>`<span class="${i<s.drawIndex?'used':''}">${face(t)}<small>${i<s.drawIndex?'已摸':'第'+(i+1)+'摸'}</small></span>`).join('')}</div><div class="challenge-river" aria-label="本次弃牌">${s.discards.length?s.discards.map(t=>face(t)).join(''):'<span>你的弃牌会留在这里</span>'}</div><div class="board-actions" role="group" aria-label="残局操作"><div class="action-copy">${s.status==='won'?'挑战成功':s.status==='failed'?'本次未完成目标':score?(goal?'已达到目标，可以胡牌':'已成和，仍需满足本题目标'):selected!=null?'已选 '+tileName(s.hand[selected]):'先选一张牌，再确认打出'}<small>${s.status==='won'?s.score.tags.map(t=>t.name).join(' · ')+' · '+s.score.fan+' 番':s.status==='failed'?'重新挑战，试试另一种选择。':'打出后立即摸下一张；杠后补牌也计一次摸牌。'}</small></div><div class="action-buttons">${actions.map((a,i)=>button(selfLabel(a),'act-'+i,a.type==='hu'&&!goal,a.type==='hu'?'gold':'')).join('')}${s.status==='playing'?button('打出并摸牌','discard',selected==null,'primary'):button('再挑战一次','retry',false,'primary')}</div></div><div class="challenge-melds">${s.melds.map(m=>`<div class="challenge-meld">${meldTiles(m).map(t=>face(t)).join('')}<small>${meldCaption(m)}</small></div>`).join('')}</div><div class="hand challenge-hand" aria-label="残局手牌">${s.hand.map((t,i)=>face(t,s.status==='playing'&&legal.includes(t)?'tile-'+i:null,selected===i)).join('')}</div></div></section><div class="challenge-tools">${button('重新挑战','retry')}${button(hint?'收起参考解法':'查看参考解法','hint')}<span>${record?`本机最佳：${record.draws} 次摸牌 · ${record.assisted?'参考后通关':'独立通关'}`:'通关记录保存在本机'}</span></div>${hint?`<section class="panel challenge-answer"><div class="panel-title">参考解法<span class="tinybadge">已查看解析</span></div><p>${p.explain}</p><small>可以有其他合法解法，通关以目标达成为准。查看解法后的通关会单独标记。</small></section>`:''}<section class="panel challenge-log"><div class="panel-title">这一次的尝试<span class="tinybadge">${s.moves.length} 步</span></div><ol>${s.log.map(t=>`<li>${esc(t)}</li>`).join('')||'<li>从现在这副手牌开始。</li>'}</ol><p class="muted">本模式是单人牌形残局：摸牌顺序公开、没有对手行牌或抢杠，不模拟完整四人血战；使用同一套定缺、幺鸡和番型判定，自摸加番，不额外计海底。</p></section></div></div>`;
  }
  async function handle(action){
    try{
      if(action.startsWith('open-')){select(action.slice(5));render();return;}
      if(action==='share'){const u=new URL(location.href);u.searchParams.delete('room');u.searchParams.set('challenge',state.id);u.hash='challenge';try{await navigator.clipboard.writeText(u.href);notice('本题链接已复制，朋友打开后从同一局面开始。');}catch{history.replaceState(null,'',u);notice('分享链接已放入地址栏，可直接复制。');}return;}
      if(action==='retry'){select(state.id);render();return;}
      if(action==='hint'){hint=!hint;if(hint)state.assisted=true;persist();render();return;}
      if(action.startsWith('tile-')){const i=Number(action.slice(5));if(!challengeDiscards(state).includes(state.hand[i]))return;selected=selected===i?null:i;render();return;}
      const wasAssisted=!!state.assisted||hint;
      const move=action==='discard'&&selected!=null?{type:'discard',tile:state.hand[selected]}:action.startsWith('act-')?challengeActions(state)[Number(action.slice(4))]:null;if(!move)return;
      state=challengeMove(state,move);state.assisted=wasAssisted;selected=null;
      if(state.status==='won'){const old=bests[state.id],record={draws:state.drawIndex,assisted:wasAssisted};if(!old||old.assisted&&!record.assisted||old.assisted===record.assisted&&record.draws<old.draws)bests[state.id]=record;}
      persist();render();
    }catch(e){notice(e.message);}
  }
  return {page,handle,sync};
}
