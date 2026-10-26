import {SUITS,tileName,recommendSwap,analyze} from './engine.js';
import {ROOM_SERVER} from './online-config.js';
const $=s=>document.querySelector(s),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const local=['localhost','127.0.0.1'].includes(location.hostname),base=local?location.origin:ROOM_SERVER;
let v=null,credential=null,ws=null,status='未入座',connected=false,selected=[],decision=null,retries=0,retryTimer,heartbeat,lastPong=0,stopped=false,toastTimer,skew=0,reviewIndex=0;
const pending=new Map(),requested=new URL(location.href).searchParams.get('room')?.toUpperCase()||'';
const key=code=>`tingchuan-seat:${base}:${code}`;
function notice(text){$('#toast').textContent=text;$('#toast').classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').classList.remove('show'),4500);}
function store(){try{sessionStorage.setItem('tingchuan-active-room',JSON.stringify({...credential,base,pending:[...pending.values()]}));localStorage.setItem(key(credential.code),JSON.stringify(credential));}catch{}}
function saved(){try{return JSON.parse(localStorage.getItem(key(requested)));}catch{return null;}}
const button=(text,action,disabled=false,cls='')=>`<button type="button" class="button ${cls}" data-action="${action}" ${disabled?'disabled':''}>${text}</button>`;
function tile(t,index=null){return `<${index==null?'span':'button type="button"'} class="tile ${selected.includes(index)&&index!=null?'selected':''}" ${index==null?'':`data-index="${index}" aria-label="${tileName(t)}" aria-pressed="${selected.includes(index)}"`} title="${tileName(t)}"><img src="./assets/tiles/${t%9+1}${['m','p','s'][Math.floor(t/9)]}.svg" alt="${tileName(t)}" draggable="false" width="40" height="56"></${index==null?'span':'button'}>`;}
function render(){
  $('#room-app').innerHTML=`<div class="shell"><aside class="sidebar"><a class="brand" href="./"><div class="brandmark">听</div><div><div class="brandname">听川</div><div class="brandlatin">TINGCHUAN</div></div></a><div class="navlabel">川 麻 研 习 室</div><nav class="nav"><a class="navbutton" href="./">▦ <span>人机练习</span></a><a class="navbutton active" href="./rooms.html" aria-current="page">♧ <span>好友同桌</span></a></nav><div class="sidebar-note"><div class="line"></div>三两好友，<br>同坐一桌川麻。</div><div class="sidebar-foot"><span><i class="statusdot"></i>好友房</span><span>v0.2.0</span></div></aside><main class="main room-main"><header class="topbar"><div class="breadcrumb">研习室 <span>/</span> 好友同桌</div><div class="connection ${connected?'online':''}" role="status">${esc(status)}</div></header><div class="content page-content"><section class="intro"><div><div class="eyebrow">SICHUAN MAHJONG · FRIENDS TABLE</div><h1>好友来，同桌坐。</h1><p>血战到底 · 换三张 · 定缺 · 自摸加番 · 五番满</p></div>${credential?`<div class="intro-actions">${button('复制邀请','invite')}${button('回到入口','home')}</div>`:''}</section>${!credential?entry():!v?`<section class="panel room-empty"><div class="room-seal">川</div><h2>${esc(status)}</h2><p>正在恢复你的座位和最新牌局。</p>${button('重新连接','reconnect')}${button('回到入口','home')}</section>`:v.phase==='lobby'?lobby():table()}<div class="room-footnote">真人与电脑可混合入座 · 操作以服务端判定为准 · 房间保留 24 小时</div></div></main></div>`;
}
function entry(){
  const previous=saved();return `<div class="room-entry"><section class="panel room-welcome"><span class="eyebrow">一桌川麻，隔屏相聚</span><h2>不够四人，也能开局。</h2><div class="welcome-tiles">${[0,9,18].map(t=>tile(t)).join('')}</div><p>创建好友房，把邀请发给朋友。<br>1–4 位真人，空位自动补齐电脑。</p><div class="room-promises"><span>01<br><b>房号入座</b></span><span>02<br><b>同步行牌</b></span><span>03<br><b>断线续局</b></span></div></section><section class="panel room-form"><h2>准备入座</h2>${!base?'<p class="room-error">联机服务尚未配置，请稍后再试。</p>':''}<label for="nickname">你的称呼</label><input id="nickname" maxlength="12" value="${esc(localStorage.getItem('tingchuan-nickname')||'牌友')}" autocomplete="nickname"><button class="button primary" data-action="create" ${!base?'disabled':''}>创建好友房 <span>→</span></button><div class="room-divider">或加入朋友的房间</div><label for="room-code">8 位房间号</label><input id="room-code" maxlength="8" placeholder="例如 ABCD2345" value="${esc(requested)}" autocomplete="off" autocapitalize="characters"><button class="button" data-action="join" ${!base?'disabled':''}>${previous?'作为新玩家加入':'加入房间'}</button>${previous?button('恢复我的座位','resume'):''}<small>断线后会自动重连。请保留当前浏览器的数据，以便恢复原座位。</small></section></div>`;
}
function seats(){return v.members.map((m,s)=>`<div class="seat-card ${s===v.seat?'mine':''}"><div class="seat-avatar">${m.bot?'机':esc(m.name[0])}</div><b>${esc(m.name)} ${s===0?'<em>房主</em>':''}${s===v.seat?'<em>你</em>':''}</b><span>${m.bot?'基础电脑 · 自动补位':`${m.online?'在线':'离线'} · ${m.ready?'已准备':'待准备'}`}</span></div>`).join('');}
function lobby(){const me=v.members[v.seat],allReady=v.members.every(m=>m.ready);return `<div class="room-lobby"><section class="panel lobby-table"><div class="lobby-title"><span class="eyebrow">等待好友入座</span><strong class="room-number">${v.code}</strong><p>邀请好友后，准备即可开始。电脑会自动补齐空位。</p></div><div class="seat-grid">${seats()}</div><div class="lobby-controls">${button(me.ready?'取消准备':'我准备好了','ready',!connected,me.ready?'':'primary')}${v.seat===0?button('开始本局','start',!connected||!allReady,'gold'):''}</div></section><aside class="panel room-help"><h2>这一桌，怎么玩？</h2><p>房主确认所有真人已准备后开局。开局后保留每位玩家的座位，想邀请新的朋友时，可重新创建房间。</p><p>每次操作有 30 秒。超时由电脑托管，可随时取消；刷新页面会自动恢复座位。</p><p>执行查花猪、查大叫与未听退税，暂不启用呼叫转移。过胡后接炮须加番，自己摸牌后解除。</p>${button('复制邀请链接','invite')}<small>邀请链接只包含房号，不包含你的座位凭证。</small></aside></div>`;}
function table(){
  const g=v.game,own=g.players[v.seat],member=v.members[v.seat];
  function player(relative,pos){const s=(v.seat+relative)%4,p=g.players[s],m=v.members[s];return `<div class="online-player ${pos} ${g.turn===s&&g.phase==='discard'?'turn':''}"><div class="seat-avatar">${p.won?'胡':m.bot?'机':esc(p.name[0])}</div><div><b>${esc(p.name)}</b><span class="score">${p.score>0?'+':''}${p.score}</span><small>${p.que<0?'尚未定缺':'缺'+SUITS[p.que]} · ${p.won?'已胡':m.bot?'电脑':m.auto?'托管':m.online?'在线':'离线'}</small><div class="online-melds">${p.melds.map(m=>tile(m.tile)+`<em>${m.type==='pong'?'碰':'杠'}</em>`).join('')}</div></div></div>`;}
  const canPick=connected&&v.pending&&!member.auto&&['swap','discard'].includes(v.phase)&&pending.size===0;
  return `<div class="online-layout"><div><div class="board-wrap"><div class="board-bar"><div class="board-bar-title">好友房 <strong>${v.code}</strong><span class="badge">血战到底</span></div><div class="board-tools"><span>${{swap:'换三张',que:'定缺',discard:'行牌',respond:'响应',ended:'已结算'}[v.phase]}</span><span class="room-timer" aria-label="本次操作剩余时间"></span></div></div><div class="felt online-felt">${player(2,'north')}${player(3,'west')}${player(1,'east')}<div class="online-backs" aria-label="对家手牌数量">${Array.from({length:g.players[(v.seat+2)%4].handCount},()=>'<span class="tileback"></span>').join('')}</div><div class="table-center"><div class="seal">听川</div><div class="seal-caption">FRIENDS AROUND THE TABLE</div></div><div class="online-rivers">${[3,2,0,1].map(r=>{const s=(v.seat+r)%4,p=g.players[s];return `<div class="river"><span class="river-label">${r===0?'你的牌河':esc(p.name)}</span>${p.discards.map((t,i)=>`<span class="${g.last?.from===s&&i===p.discards.length-1?'latest-discard':''}">${tile(t)}</span>`).join('')}</div>`;}).join('')}</div><div class="wall-counter">牌墙余 <strong>${g.remaining}</strong> 张</div><div class="own-zone"><div class="own-info"><div class="own-player"><span class="avatar ${g.turn===v.seat?'active':''}">${own.won?'胡':'你'}</span><b>${esc(own.name)}</b><span class="smalltag">${own.que<0?'尚未定缺':'定缺 · '+SUITS[own.que]}</span><span class="smalltag">${own.score>0?'+':''}${own.score} 分</span></div><span>${member.auto?'电脑托管中':!connected?'连接恢复中':''}</span></div><div class="melds">${own.melds.map(m=>`<div class="meld">${Array(m.type==='pong'?3:4).fill(m.tile).map(t=>tile(t)).join('')}</div>`).join('')}</div><div class="hand">${own.hand.map((t,i)=>tile(t,canPick?i:null)).join('')}</div></div><span class="session-marker">换三张 / 定缺 / 自摸加番 / 五番满</span></div>${actions()}</div><p class="sub-board">换三张、定缺同时提交；胡牌优先，支持一炮多响。断线或超时托管不会重置牌局。</p>${v.phase==='ended'?settlement():''}</div><aside class="right-column"><section class="panel room-help"><div class="panel-title">同桌动态<span class="tinybadge">实时同步</span></div><div class="online-members">${v.members.map((m,s)=>`<div><b>${esc(m.name)}${s===v.seat?' · 你':''}</b><span>${m.bot?'电脑':m.auto?'托管':m.online?'在线':'断线'}</span></div>`).join('')}</div>${v.phase!=='ended'?button(member.auto?'取消电脑托管':'交给电脑托管','auto',!connected):''}<p>对战时隐藏牌效建议。结算后可以查看自己的每一步出牌，避免影响同桌对局。</p>${own.passedHuFan!=null&&!own.won?`<p class="room-lock">过手胡：接炮需高于 ${own.passedHuFan} 番，自己摸牌后解除。</p>`:''}</section><section class="panel room-help"><div class="panel-title">本局记录</div><ol class="online-log">${g.recent.map(e=>`<li>${esc(e.text)}</li>`).join('')}</ol></section></aside></div>`;
}
function actions(){
  const g=v.game,disabled=!connected||pending.size>0,auto=v.members[v.seat].auto;let text='',sub='',buttons='';
  if(v.phase==='ended'){text='本局已结算';sub='重新准备后，房主可开始下一局';buttons=button(v.members[v.seat].ready?'已准备 · 取消':'准备下一局','ready',disabled)+ (v.seat===0?button('再来一局','start',disabled||!v.members.every(m=>m.ready),'primary'):'');}
  else if(!connected){text='连接已中断，正在重连';sub='恢复连接后会同步最新牌局';}
  else if(auto){text='电脑正在替你行牌';sub='取消托管后，从下一次可操作的决定继续';buttons=button('取消托管','auto',disabled,'primary');}
  else if(!v.pending){text=v.submitted?'已提交，等待其他牌友':'等待同桌行牌';sub=g.players[v.seat].won?'你已胡牌，可以继续观察本局':'每家独立提交，服务端统一判定';}
  else if(v.phase==='swap'){text='选择同一花色的三张牌';sub=`${['','顺时针','对家','逆时针'][g.swapDirection]}交换 · 已选 ${selected.length}/3`;buttons=button('帮我选三张','suggest-swap',disabled)+button('确认换牌','swap',disabled||selected.length!==3,'primary');}
  else if(v.phase==='que'){text='选一门花色作为定缺';sub='四家定缺完成后，庄家先出牌';buttons=SUITS.map((s,i)=>button('缺'+s,'que-'+i,disabled,'primary')).join('');}
  else if(v.phase==='respond'){text=`${esc(g.players[g.last.from].name)}${g.last.rob?'补杠':'打出'} ${tileName(g.last.tile)}`;sub='过胡后，同番或更低番不能接炮，自己摸牌后解除';buttons=button('过','respond-pass',disabled)+v.options.responses.map(a=>button({hu:'胡牌',pong:'碰',kong:'杠'}[a],'respond-'+a,disabled,a==='hu'?'gold':'primary')).join('');}
  else if(v.phase==='discard'){text=selected.length?'已选 '+tileName(g.players[v.seat].hand[selected[0]]):'轮到你出牌';sub='缺门未清时，必须优先打缺门';buttons=v.options.self.map(a=>button(a.type==='hu'?'自摸胡':`${a.type==='ankong'?'暗杠':'补杠'}${tileName(a.tile)}`,`self-${a.type}-${a.tile??''}`,disabled,a.type==='hu'?'gold':'')).join('')+button('打出','discard',disabled||selected.length!==1,'primary');}
  return `<div class="board-actions"><div class="action-copy">${text}<small>${sub}</small></div><div class="action-buttons">${buttons}</div></div>`;
}
function settlement(){
  const reviews=v.review||[],r=reviews[Math.min(reviewIndex,reviews.length-1)],analysis=r?analyze(r.obs):[];
  return `<section class="panel online-settlement"><div class="panel-title">本局结算<span class="tinybadge">服务端结算</span></div><div class="settlement-players">${v.game.players.map(p=>`<div><b>${esc(p.name)}</b><strong class="${p.score<0?'negative':''}">${p.score>0?'+':''}${p.score}</strong><small>${p.won?`${p.win.fan}番 · ${p.win.self?'自摸':'接炮'}`:'未胡牌'}</small></div>`).join('')}</div><details><summary>查看逐笔结算</summary><div class="ledger-list">${v.game.ledger.map(x=>`<p>${esc(v.game.players[x.from].name)} → ${esc(v.game.players[x.to].name)}：${x.amount} 分 · ${esc(x.reason)}${x.refunded?'（已退税）':''}</p>`).join('')||'<p>本局无收支</p>'}</div></details><div class="panel-title">我的逐步复盘 <span>${reviews.length} 次出牌</span></div>${r?`<div class="review-controls">${button('上一手','review-prev',reviewIndex===0)}<span>第 ${reviewIndex+1} / ${reviews.length} 手</span>${button('下一手','review-next',reviewIndex>=reviews.length-1)}${button('导出我的复盘','export')}</div><div class="review-hand">${r.obs.hand.map(t=>tile(t)).join('')}</div><p>你打出 ${tileName(r.chosen)}；牌效建议 ${tileName(analysis[0].tile)}。</p><div class="review-candidates">${analysis.slice(0,5).map(a=>`<span>${tileName(a.tile)} · ${a.shanten===0?'听牌':a.shanten+'向听'} · 进张 ${a.ukeire}</span>`).join('')}</div><small>仅按当时自己的手牌与公开信息评估向听和进张，尚不等于长期得分最优。</small>`:'<p>你本局没有出牌记录。</p>'}</section>`;
}
function updateClock(){if(!v)return;const el=$('.room-timer');if(el)el.textContent=v.deadline&&v.phase!=='ended'?Math.max(0,Math.ceil((v.deadline-(Date.now()+skew))/1000))+' 秒':'';}
setInterval(updateClock,500);
function connect(){
  if(!credential)return;stopped=false;connected=false;clearTimeout(retryTimer);clearInterval(heartbeat);status=retries?'正在重连…':'连接牌桌…';render();
  const url=new URL(`${base}/rooms/${credential.code}/socket`);url.protocol=url.protocol==='https:'?'wss:':'ws:';const socket=new WebSocket(url);ws=socket;let authenticated=false;
  socket.onopen=()=>{socket.send(JSON.stringify({type:'auth',token:credential.token}));lastPong=Date.now();heartbeat=setInterval(()=>{if(Date.now()-lastPong>50000){socket.close();return;}if(socket.readyState===1)socket.send('ping');},20000);};
  socket.onmessage=event=>{
    if(ws!==socket)return;lastPong=Date.now();if(event.data==='pong')return;
    let m;try{m=JSON.parse(event.data);}catch{return;}
    if(m.type==='state'){
      if(m.protocol!==1){notice('版本已更新，请刷新网页。');stopped=true;socket.close();return;}
      if(v&&m.revision<v.revision)return;
      v=m;connected=true;status='已连接 · 实时同步';retries=0;skew=m.serverTime-Date.now();
      if(decision!==m.decision){decision=m.decision;selected=[];}
      if(!authenticated){authenticated=true;for(const c of pending.values())socket.send(JSON.stringify(c));}
      render();updateClock();
    }else if(m.type==='ack'||m.type==='error'){
      if(m.id){pending.delete(m.id);store();}if(m.type==='error')notice(m.message);render();
    }
  };
  socket.onclose=event=>{
    if(ws!==socket)return;clearInterval(heartbeat);connected=false;
    if([1008,4001,4004].includes(event.code)){stopped=true;status=event.code===4001?'此座位已在另一页面打开':event.reason||'座位无法恢复';notice(status);}
    else status='连接中断 · 自动重连中';render();
    if(!stopped)retryTimer=setTimeout(connect,Math.min(10000,600*2**Math.min(5,retries++))+Math.random()*300);
  };
  socket.onerror=()=>{};
}
function command(action,extra={}){
  if(!connected||ws?.readyState!==1){notice('正在恢复连接，请稍候。');return;}
  if(pending.size){notice('上一操作正在确认。');return;}
  const c={type:'command',id:crypto.randomUUID(),decision:v.decision,action,...extra};pending.set(c.id,c);store();ws.send(JSON.stringify(c));render();
}
async function enter(create){
  const name=$('#nickname').value.trim()||'牌友',code=$('#room-code').value.trim().toUpperCase();
  if(!create&&!/^[A-Z2-9]{8}$/.test(code)){notice('请输入正确的 8 位房号。');return;}
  const btn=$(`[data-action="${create?'create':'join'}"]`);btn.disabled=true;
  try{
    const response=await fetch(`${base}${create?'/rooms':'/rooms/'+code+'/join'}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name}),signal:AbortSignal.timeout(15000)});const body=await response.json();if(!response.ok)throw Error(body.error||'暂时无法入座');
    credential=body;localStorage.setItem('tingchuan-nickname',name);pending.clear();store();history.replaceState(null,'',`?room=${body.code}`);connect();
  }catch(e){notice(e.message==='Failed to fetch'?'暂时无法连接房间服务，请稍后重试。':e.message);btn.disabled=false;}
}
document.addEventListener('click',async event=>{
  const pick=event.target.closest('[data-index]');if(pick){const i=Number(pick.dataset.index);if(v.phase==='swap')selected=selected.includes(i)?selected.filter(x=>x!==i):selected.length<3?[...selected,i]:selected;else selected=[i];render();updateClock();return;}
  const action=event.target.closest('[data-action]')?.dataset.action;if(!action)return;
  if(action==='create'||action==='join')return enter(action==='create');
  if(action==='resume'){credential=saved();if(credential){store();connect();}return;}
  if(action==='reconnect'){retries=0;ws?.close();connect();return;}
  if(action==='home'){stopped=true;clearTimeout(retryTimer);clearInterval(heartbeat);ws?.close();ws=null;credential=null;v=null;selected=[];pending.clear();sessionStorage.removeItem('tingchuan-active-room');history.replaceState(null,'','rooms.html');render();return;}
  if(action==='invite'){const invite=new URL('rooms.html',location.href);invite.searchParams.set('room',credential.code);try{await navigator.clipboard.writeText(invite.href);notice('邀请链接已复制，发给朋友即可入座。');}catch{notice('房号 '+credential.code+'，可从地址栏复制邀请链接。');}return;}
  if(action==='suggest-swap'){selected=recommendSwap(v.game.players[v.seat].hand);render();return;}
  if(action==='ready'||action==='start')return command(action);
  if(action==='auto')return command('auto',{value:!v.members[v.seat].auto});
  if(action==='swap')return command('swap',{indices:selected});
  if(action.startsWith('que-'))return command('que',{suit:Number(action.slice(4))});
  if(action==='discard')return command('discard',{tile:v.game.players[v.seat].hand[selected[0]]});
  if(action.startsWith('respond-'))return command('respond',{choice:action.slice(8)});
  if(action.startsWith('self-')){const [,choice,t]=action.split('-');return command('self',{choice,...(t!==''?{tile:Number(t)}:{})});}
  if(action==='review-prev'||action==='review-next'){reviewIndex=Math.max(0,Math.min(v.review.length-1,reviewIndex+(action==='review-next'?1:-1)));render();return;}
  if(action==='export'){
    const data={format:'tingchuan-room-review-v1',room:credential.code,player:v.members[v.seat].name,rules:v.rules,review:v.review,ledger:v.game.ledger};
    const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=`听川-我的联机复盘-${credential.code}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
});
window.addEventListener('online',()=>{if(credential&&!connected&&!stopped)connect();});
try{const previous=JSON.parse(sessionStorage.getItem('tingchuan-active-room'));if(previous?.base===base&&(!requested||requested===previous.code)){credential={code:previous.code,token:previous.token,seat:previous.seat};for(const c of previous.pending||[])pending.set(c.id,c);}}catch{}
if(credential)connect();else render();
