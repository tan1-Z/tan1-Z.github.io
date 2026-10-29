import {BOT_STYLES,normalizeBotStyles,botStyleName} from './bot-policy.js?v=0.6.0';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export {botStyleName};
export function botPicker(styles,{names=['你','竹隐','云岫','松风'],seats=[1,2,3],disabled=false,prefix='bot-style'}={}){
  const ids=normalizeBotStyles(styles);
  if(!seats.length)return '<p class="bot-note">本桌均为真人，超时托管使用均衡型。</p>';
  return `<fieldset class="bot-picker"><legend>电脑性格</legend><div class="bot-picker-grid">${seats.map(s=>`<label for="${prefix}-${s}">${esc(names[s])}<select id="${prefix}-${s}" data-bot-seat="${s}" aria-label="${esc(names[s])}的电脑性格" ${disabled?'disabled':''}>${Object.entries(BOT_STYLES).map(([id,info])=>`<option value="${id}" ${ids[s]===id?'selected':''}>${info.name}</option>`).join('')}</select><small>${BOT_STYLES[ids[s]].description}</small></label>`).join('')}</div><p class="bot-note">性格影响取舍，分析深度一致。所有性格都优先合法胡牌；防守风险是估计，不保证安全。设置在新局生效，固定局数比赛中保持不变。</p></fieldset>`;
}
export function readBotPicker(root,fallback){const ids=normalizeBotStyles(fallback);for(const select of root.querySelectorAll('[data-bot-seat]'))ids[Number(select.dataset.botSeat)]=select.value;return normalizeBotStyles(ids);}
export function botRoster(game){return `<section class="bot-roster" aria-label="本桌电脑性格"><div><strong>本桌电脑</strong><span>${game.players.slice(1).map((p,i)=>`${esc(p.name)} · ${botStyleName(game.botStyles?.[i+1])}`).join('　/　')}</span></div><button type="button" class="button" data-action="new">设置下一局</button></section>`;}
