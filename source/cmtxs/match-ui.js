const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const signed=n=>(n>0?'+':'')+n;
const name=(m,members,s)=>m.results.at(-1)?.players[s]?.name||members[s]?.name||'牌友';
export const matchName=m=>m.rounds?`${m.rounds} 局小赛`:'自由对局';
export function reportText(m,members,code){
  const n=s=>name(m,members,s);
  const lines=[`听川 · ${matchName(m)}${m.completed?'总战报':'阶段战报'}`,`房间 ${code} · 已结算 ${m.played}${m.rounds?' / '+m.rounds:''} 局`,...m.totals.map((total,s)=>({s,total})).sort((a,b)=>b.total-a.total).map(x=>`第${m.ranks[x.s]}名 ${n(x.s)}：${signed(x.total)}分 · 胡牌${m.wins[x.s]}次 · 自摸${m.selfs[x.s]}次 · 点炮送胡${m.fires[x.s]}次`), '',...m.results.map(r=>`第${r.number}局：${r.players.map(p=>p.name+' '+signed(p.score)).join(' / ')}`)];
  if(m.biggest)lines.push(`最大胡牌：${n(m.biggest.seat)}，第${m.biggest.round}局 ${m.biggest.fan}番，胡牌收入${m.biggest.points}分`);
  if(m.mostFires.length)lines.push(`最多点炮送胡：${m.mostFires.map(n).join('、')}，${m.fires[m.mostFires[0]]}次（一炮多响按胡牌人数计）`);
  if(m.comeback)lines.push(`最佳逆转：${n(m.comeback.seat)}，第${m.comeback.round}局追平或反超此前领先者，开局落后${m.comeback.gap}分`);
  lines.push('累计分含杠分、查叫与退税；并列同分共享名次。');return lines.join('\n');
}
export function renderMatch(v,{full=false}={}){
  const m=v.match;if(!m)return '';const n=s=>esc(name(m,v.members,s));
  const ordered=m.totals.map((total,s)=>({s,total})).sort((a,b)=>b.total-a.total);
  return `<section class="panel match-panel ${full?'match-full':''}" aria-label="本场累计战报"><div class="panel-title">${m.completed?'本场最终战报':'本场累计'}<span class="tinybadge">${matchName(m)} · ${m.played}${m.rounds?' / '+m.rounds:''} 局已结算</span></div><div class="match-standings">${ordered.map(({s,total})=>`<div class="${s===v.seat?'mine':''}"><small>${m.played?'第 '+m.ranks[s]+' 名':m.started?'首局进行中':'待开局'}</small><b>${n(s)}${v.members[s]?.bot?'<em>电脑</em>':''}</b><strong class="${total<0?'negative':''}">${signed(total)}</strong><span>胡 ${m.wins[s]} · 自摸 ${m.selfs[s]}</span></div>`).join('')}</div>${full&&m.played?`<div class="match-awards">${m.biggest?`<p><b>最大胡牌</b>${n(m.biggest.seat)} · 第 ${m.biggest.round} 局 ${m.biggest.fan} 番 · 胡牌收入 ${m.biggest.points} 分</p>`:''}${m.mostFires.length?`<p><b>最多点炮送胡</b>${m.mostFires.map(n).join('、')} · ${m.fires[m.mostFires[0]]} 次</p>`:''}${m.comeback?`<p><b>最佳逆转</b>${n(m.comeback.seat)} · 第 ${m.comeback.round} 局从落后 ${m.comeback.gap} 分追平或反超</p>`:''}</div><details class="match-rounds"><summary>逐局积分与胡牌记录</summary><div class="match-table-scroll"><table><thead><tr><th>局数</th>${[0,1,2,3].map(s=>`<th>${n(s)}</th>`).join('')}</tr></thead><tbody>${m.results.map(r=>`<tr><th>第 ${r.number} 局</th>${r.players.map(p=>`<td><b>${signed(p.score)}</b><small>${p.won?`${p.fan}番 ${p.self?'自摸':'接炮'}`:'未胡'}</small></td>`).join('')}</tr>`).join('')}</tbody></table></div></details><div class="match-report-actions"><button type="button" class="button" data-action="match-copy">复制战报</button><button type="button" class="button" data-action="match-export">下载战报</button></div><small class="match-note">累计分包含杠分、查叫与退税。点炮送胡按胡牌人数计；同分并列。${m.completed?'本场已结束，新开一场后重新准备。':'每局结算后更新，下一局沿用当前累计分。'}</small>`:''}</section>`;
}
