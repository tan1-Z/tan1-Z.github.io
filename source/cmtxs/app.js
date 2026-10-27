import { VERSION,RULES,SUITS,tileName,sorted,counts,createGame,recommendSwap,recommendQue,swapTiles,setQue,observe,analyze,shanten,legalDiscards,selfActions,selfAction,discard,resolveResponse,aiStep,scoreHand,isWinning,assertGame,isCompatibleVersion,restoreGame,refreshDecision,normalizeRules,modeName,meldTiles,isQueTile,clearQue,waits } from './engine.js?v=0.3.0';
import {variantPicker,variantHint,claimLabel,selfLabel,meldCaption} from './variants-ui.js?v=0.3.0';

const $=s=>document.querySelector(s);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
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
const icon=(name)=>`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name]||icons.info}</svg>`;
const btn=(label,action,cls='',disabled=false)=>`<button type="button" class="button ${cls}" data-action="${action}" ${disabled?'disabled':''}>${label}</button>`;

function tileSVG(t){
  const name=(t%9+1)+['m','p','s'][Math.floor(t/9)];
  return '<img src="./assets/tiles/'+name+'.svg" alt="'+tileName(t)+'" draggable="false" width="40" height="56">';
}

function tile(t,{index,selected=false,forced=false,suggested=false,small=false,action='',last=false,joker=false}={}){
  const interactive=index!==undefined||action;
  return `<${interactive?'button':'span'} ${interactive?'type="button"':''} class="tile ${selected?'selected':''} ${forced?'forced':''} ${suggested?'suggested':''} ${last?'last':''} ${joker?'joker':''}" ${index!==undefined?`data-tile-index="${index}"`:''} ${action?`data-action="${action}"`:''} title="${tileName(t)}" ${interactive?`aria-label="${tileName(t)}${selected?'，已选择':''}" aria-pressed="${selected}"`:''}>${tileSVG(t)}${joker?'<b class="joker-mark">百搭</b>':''}</${interactive?'button':'span'}>`;
}

const KEY='tingchuan-v1',ARCHIVE='tingchuan-archive-v1';
let game,archive=[];
try{const saved=JSON.parse(localStorage.getItem(KEY));game=isCompatibleVersion(saved?.version)?restoreGame(saved):null;archive=JSON.parse(localStorage.getItem(ARCHIVE)||'[]').filter(g=>isCompatibleVersion(g?.version)).map(restoreGame);}catch{}
if(!game)game=createGame();
let page='live',selected=[],speed=650,paused=false,sound=false,timer,toastTimer,reviewIndex=0,reviewSeed=null,settledSeed=null,trainIndex=0,trainChoice=null;
let analysisCache={key:'',data:[]};
function save(){try{localStorage.setItem(KEY,JSON.stringify(game));localStorage.setItem(ARCHIVE,JSON.stringify(archive.slice(0,3)));}catch{toast('本机存储已满，可先导出牌谱再清理历史。');}}
function toast(message){$('#toast').textContent=message;$('#toast').classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').classList.remove('show'),3300);}
function playSound(){if(!sound)return;try{const ctx=new(window.AudioContext||window.webkitAudioContext)(),o=ctx.createOscillator(),gain=ctx.createGain();o.type='triangle';o.frequency.setValueAtTime(650,ctx.currentTime);o.frequency.exponentialRampToValueAtTime(250,ctx.currentTime+.06);gain.gain.setValueAtTime(.04,ctx.currentTime);gain.gain.exponentialRampToValueAtTime(.001,ctx.currentTime+.1);o.connect(gain);gain.connect(ctx.destination);o.start();o.stop(ctx.currentTime+.1);o.onended=()=>ctx.close();}catch{}}
function storeArchive(g){if(!g.events.length)return;archive=[structuredClone(g),...archive.filter(a=>a.seed!==g.seed)].slice(0,3);save();}
function currentAnalysis(){
  if(game.phase!=='discard'||game.turn!==0||game.players[0].won)return [];
  const obs=observe(game,0),key=JSON.stringify(obs);
  if(analysisCache.key!==key)analysisCache={key,data:analyze(obs)};
  return analysisCache.data;
}
function phaseText(g){return {swap:'换三张',que:'定缺',discard:g.turn===0?'轮到你出牌':'电脑行牌中',respond:g.decision?'等待你的响应':'响应判定中',ended:'本局已结束'}[g.phase];}
function shantenText(n){return n<0?'已成和':n===0?'已听牌':`${n}向听`;}
function playerShanten(p,rules=game.rules){return shanten(p.hand.filter(t=>!isQueTile(t,p.que,rules)),p.melds.length,rules);}
function reviewGrade(review){const best=review.analysis[0],actual=review.analysis.find(x=>x.tile===review.chosen);if(!actual||!best)return {label:'已记录',good:true};if(actual.shanten===best.shanten&&actual.ukeire===best.ukeire)return {label:'牌效推荐',good:true};return {label:actual.shanten>best.shanten?'向听退步':'进张较少',good:false};}
function latestReview(){return [...game.events].reverse().find(e=>e.review)?.review;}
function render(){
  refreshDecision(game);
  const titles={live:'人机练习',review:'逐步复盘',train:'何切训练',rules:'规则手册'};
  $('#app').innerHTML=`<div class="shell"><aside class="sidebar"><div class="brand"><div class="brandmark">听</div><div><div class="brandname">听川</div><div class="brandlatin">TINGCHUAN</div></div></div><div class="navlabel">川 麻 研 习 室</div><nav class="nav" aria-label="主导航">${[['live','table','人机练习'],['review','review','逐步复盘'],['train','train','何切训练'],['rules','rules','规则手册']].map(([id,ico,label])=>`<button type="button" class="navbutton ${page===id?'active':''}" data-page="${id}" ${page===id?'aria-current="page"':''}>${icon(ico)}<span>${label}</span></button>`).join('')}<a class="navbutton" href="./rooms.html" style="text-decoration:none">${icon('users')}<span>好友同桌</span></a></nav><div class="sidebar-note"><div class="line"></div>一张一弛，<br>慢慢听见牌的答案。</div><div class="sidebar-foot"><span><i class="statusdot"></i>本地研习</span><span>v${VERSION}</span></div></aside><main class="main"><header class="topbar"><div class="breadcrumb"><span>研习室</span><span>/</span><strong>${titles[page]}</strong></div><div class="mobile-brand">听川<small>川麻研习室</small></div><div class="top-actions"><span class="rule-pill">${icon('leaf')}血战到底 · 自摸加番 · 五番满</span><span class="divider"></span><button type="button" class="iconbutton" data-action="sound" title="${sound?'关闭':'开启'}音效" aria-label="${sound?'关闭':'开启'}音效">${icon(sound?'sound':'mute')}</button><button type="button" class="iconbutton" data-action="help" aria-label="操作帮助">${icon('help')}</button><span class="iconbutton user-avatar" title="本地玩家">研</span></div></header><div class="content">${page==='live'?livePage():page==='review'?reviewPage():page==='train'?trainPage():rulesPage()}<footer class="footer"><span class="wordmark">听川 · 以牌会意</span><span>牌效基线 v${VERSION} · 对局仅保存在本机</span></footer></div></main></div>`;
  if(page==='live')schedule();
}
function intro(eyebrow,title,desc,actions=''){return `<section class="intro"><div><div class="eyebrow">${eyebrow}</div><h1>${title}</h1><p>${desc}</p></div><div class="intro-actions">${actions}</div></section>`;}
function livePage(){
  const n=game.events.filter(e=>e.review).length;
  return `${intro('SICHUAN MAHJONG · STUDY ROOM','每一手，都有依据。','在一局川麻里，练习判断，也看见每个选择的理由。',btn(icon('download')+'牌谱','export')+btn(icon('plus')+'新开一局','new','primary'))}<div class="game-layout"><div><div class="board-wrap"><div class="board-bar"><div class="board-bar-title">${icon('table').replace('<svg','<svg style="width:15px;height:15px;color:#75846b"')}<span>人机练习</span><span class="badge">${modeName(game.rules)}</span></div><div class="board-tools"><span>${phaseText(game)}</span><select id="speed" aria-label="电脑行牌速度"><option value="650" ${speed===650?'selected':''}>常速</option><option value="220" ${speed===220?'selected':''}>快速</option><option value="1100" ${speed===1100?'selected':''}>慢速</option></select></div></div>${board(game)}${liveActions()}</div><div class="sub-board">${icon('info')}<span>先选牌，再确认。缺门未清时需先打缺门牌；电脑使用与你相同的可见信息。</span><span class="keyboard-hint"><span class="kbd">Enter</span> 确认出牌</span></div></div><aside class="right-column">${coachPanel()}<section class="panel secondary-panel"><div class="panel-title">本局足迹<span class="tinybadge">自动记录</span></div><div class="session-metrics"><div class="session-metric"><strong>${n.toString().padStart(2,'0')}</strong><span>已记录出牌</span></div><div class="session-metric"><strong>${game.winners.length}<span style="display:inline;font-size:12px"> / 3</span></strong><span>已胡牌玩家</span></div><div class="session-metric"><strong class="${game.players[0].score<0?'negative':''}">${game.players[0].score>0?'+':''}${game.players[0].score}</strong><span>你的当前分数</span></div></div><div class="session-progress"><span style="width:${Math.max(2,(55-game.wall.length)/55*100)}%"></span></div><div class="session-caption">${game.phase==='ended'?'本局已收录，去复盘看看关键选择。':'每一次出牌，都为下一次判断留下依据。'}</div><button type="button" class="coach-link" data-page="review"><span>查看本局复盘</span>${icon('arrow')}</button></section><section class="training-banner"><div><strong>把判断，练成直觉。</strong><p>何切小练习 · 从牌效开始</p></div><button type="button" class="arrow" data-page="train" aria-label="进入何切训练">↗</button></section></aside></div>`;
}
function board(g,{review=false,highlight=null}={}){
  const own=g.players[0],rules=normalizeRules(g.rules);
  const opponent=(s,pos)=>{const p=g.players[s];return `<div class="opponent ${pos}"><div class="avatar ${!review&&g.turn===s&&g.phase==='discard'?'active':''} ${p.won?'won':''}">${p.won?'胡':p.name[0]}</div><div><div class="playername">${p.name} <span class="opponent-score">${p.score>0?'+':''}${p.score}</span></div><div class="playerdetail">${p.que<0?'待定缺':'缺'+SUITS[p.que]}${p.won?' · 已胡':' · 基础电脑'}</div>${p.melds.length?`<div class="opponent-meld">${p.melds.map(m=>`<span title="${meldCaption(m)}">${tile(m.tile)}<small>${meldCaption(m)}</small></span>`).join('')}</div>`:''}</div></div>`;};
  return `<div class="felt"><span class="felt-corner">${review?'出牌前局面':'东家坐庄'} · ${g.phase==='swap'?'换三张':g.phase==='que'?'定缺':'血战到底'}</span>${opponent(2,'top')}${opponent(3,'left')}${opponent(1,'right')}<div class="backs" aria-label="对家暗手">${Array.from({length:Math.min(g.players[2].hand.length,13)},()=>'<span class="tileback"></span>').join('')}</div><div class="table-center"><div class="seal">听川</div><div class="seal-caption">THE ART OF EVERY CHOICE</div></div><div class="river-area">${[3,2,0,1].map(s=>`<div class="river"><span class="river-label">${s===0?'你的牌河':g.players[s].name}</span>${g.players[s].discards.slice(-18).map((t,i,arr)=>tile(t,{last:!review&&i===arr.length-1&&g.last?.from===s})).join('')}</div>`).join('')}</div><div class="wall-counter">牌墙余 <strong>${g.wall.length}</strong> 张</div><div class="own-zone">${own.melds.length?`<div class="melds">${own.melds.map(m=>`<div class="meld">${meldTiles(m).map(t=>tile(t,{joker:rules.joker&&t===18})).join('')}<small class="meld-caption">${meldCaption(m)}</small></div>`).join('')}</div>`:''}<div class="own-info"><div class="own-player"><span class="avatar ${!review&&g.turn===0&&g.phase==='discard'?'active':''}">${own.won?'胡':'你'}</span><span>研习者</span><span class="smalltag">${own.que<0?'尚未定缺':'定缺 · '+SUITS[own.que]}</span></div><span class="own-state">${review?'按当时信息进行牌效分析':own.won?'你已胡牌，可继续观察本局':g.phase==='swap'?`同色三张 · ${['','顺时针','对家','逆时针'][g.swapDirection]}交换`:g.phase==='que'?'选择一门花色作为定缺':g.phase==='discard'&&g.turn===0?shantenText(playerShanten(own,rules)):''}</span></div><div class="hand">${own.hand.map((t,i)=>tile(t,{index:!review&&(g.phase==='swap'||g.phase==='discard'&&g.turn===0&&!own.won&&legalDiscards(g,0).includes(t))?i:undefined,selected:!review&&selected.includes(i),forced:isQueTile(t,own.que,rules),joker:rules.joker&&t===18, suggested:review&&t===highlight})).join('')}</div></div>${g.phase==='ended'?'<div class="ended-tag">一局有终，研习无尽。</div>':''}${!review&&paused?`<div class="paused-overlay"><span>歇一歇，再落牌。</span>${btn(icon('play')+'继续对局','pause','gold')}</div>`:''}<span class="session-marker">${review?'REVIEW · PLAYER INFORMATION ONLY':modeName(rules)+' / 自摸加番 / 五番满'}</span></div>`;
}
function liveActions(){
  const p=game.players[0];let text='',sub='',buttons='';
  if(paused){text='对局已暂停';sub='继续后再出牌';buttons=btn(icon('play')+'继续','pause','primary');}
  else if(game.phase==='swap'){text='选三张，换个好开局。';sub=`已选择 ${selected.length} / 3 张，须为同一花色`;buttons=btn('帮我选','auto-swap','small')+btn('确认换牌','swap','primary',selected.length!==3);}
  else if(game.phase==='que'){const q=recommendQue(p.hand,game.rules);text='这一局，定缺哪一门？';sub='缺门牌需要优先打完';buttons=`<div class="que-buttons">${[0,1,2].map(s=>`<button type="button" class="que-button ${s===q?'recommended':''}" data-action="que-${s}">缺${SUITS[s]}${s===q?' · 建议':''}</button>`).join('')}</div>`;}
  else if(game.phase==='ended'){text='本局结算已完成';sub='每一步都已留在牌谱里';buttons=btn('查看结算','settlement','small')+btn('逐步复盘','review-current','primary');}
  else if(p.won){text='你已胡牌，看看接下来的牌局。';sub='血战到底，其他玩家继续';buttons=btn(icon(paused?'play':'pause')+(paused?'继续':'暂停'),'pause','small')+btn('复盘我的出牌','review-current');}
  else if(game.phase==='respond'&&game.decision){text=`${game.players[game.decision.from].name}${game.decision.rob?'补杠':'打出'}${tileName(game.decision.tile)}`;sub=game.decision.rob?'可以选择抢杠胡':'胡牌优先；支持一炮多响';buttons=btn('过','pass','small')+game.decision.actions.map(a=>btn(claimLabel(a),'claim-'+a,a==='hu'?'gold':'primary')).join('');}
  else if(game.turn===0&&game.phase==='discard'){
    text=selected.length?`已选 ${tileName(p.hand[selected[0]])}`:'轮到你了，慢慢想。';sub=selected.length?'确认后打出，可在右侧比较其他选择':'点击手牌选择，再确认出牌';
    const acts=selfActions(game);buttons=acts.map(a=>btn(selfLabel(a),`self-${a.type}-${a.tile??''}-${a.jokers??0}`,a.type==='hu'?'gold':'small')).join('')+btn('打出','discard','primary',selected.length!==1);
  }else{text=game.message;sub='电脑仅根据自己可见的信息决策';buttons=btn(icon(paused?'play':'pause')+(paused?'继续':'暂停'),'pause','small');}
  if(p.passedHuFan!=null&&!p.won&&game.phase!=='ended')sub=`过手胡：接炮需高于 ${p.passedHuFan} 番，自己摸牌后解除。${sub}`;
  return `<div class="board-actions"><div class="action-copy">${text}<small>${sub}</small></div><div class="action-buttons">${buttons}</div></div>`;
}
function coachPanel(review=null){
  const list=review?.analysis||currentAnalysis(),p=game.players[0];
  if(!list.length){
    const stage=game.phase==='swap'?0:game.phase==='que'?1:2;
    const title=game.phase==='ended'?'好牌局，值得再看。':p.won?'胡牌之后，回看判断。':stage===0?'好开局，从取舍开始。':stage===1?'定一门，留一条路。':'等下一手，新的判断。';
    const desc=stage===0?'先换三张，再定缺。第一版电脑以牌效为基线，陪你完整练习一局。':stage===1?`你当前${SUITS[recommendQue(p.hand,game.rules)]}牌较少，可以优先考虑；定缺也要兼顾已有搭子。`:'轮到你出牌时，这里会比较候选牌的向听数与未见有效进张。';
    return `<section class="panel coach-panel"><div class="panel-title">随手研习<span class="tinybadge">牌效基线</span></div><div class="coach-empty-icon">${[2,12,19].map(t=>tile(t)).join('')}</div><h2 class="coach-title">${title}</h2><p class="coach-desc">${desc}</p><div class="mini-steps">${[...(game.rules.swap?[['换三张','同色三张，重新组合']]:[]),['定缺','先清一门，专注成牌'],['行牌研习','比较牌效，记录每一步']].map(([title,desc],i)=>`<div class="mini-step ${stage===i?'current':''}"><b>0${i+1}</b><span>${title}<br><span style="font-size:9px;opacity:.7;line-height:2.1">${desc}</span></span></div>`).join('')}</div><div class="info-footnote">${icon('info')}<span>当前提供牌效建议，尚不评估放炮概率与长期得分。</span></div></section>`;
  }
  return analysisPanel(list,review);
}
function analysisPanel(list,review=null){
  const best=list[0],selectedTile=review?review.chosen:selected.length?game.players[0].hand[selected[0]]:null,actual=list.find(a=>a.tile===selectedTile);
  const desc=best.forced?'缺门未清，先比较缺门牌的打出顺序。':'在合法出牌中，先比较向听数，再比较未见有效进张。';
  let explanation=best.shanten===0?`打出后可听 ${best.improves.length} 种牌，共 ${best.ukeire} 张未见牌。`:best.shanten<0?'当前手牌已成和，请优先检查胡牌操作。':`打出后为${best.shanten}向听，有 ${best.improves.length} 种牌能让手牌进一步成形。`;
  return `<section class="panel coach-panel"><div class="panel-title">${review?'这一步的选择':'这一手，怎么选？'}<span class="tinybadge">牌效基线</span></div><p class="coach-desc">${desc}</p><div class="coach-hero"><div class="coach-hero-label">${icon('spark')}当前牌效推荐</div><div class="coach-choice">${tile(best.tile)}<div><strong>打 ${tileName(best.tile)}</strong><small>${best.forced?'优先清缺门':'向听优先 · 进张次之'}</small></div></div><div class="metrics"><div class="metric"><span>打出后向听</span><strong>${best.shanten===0?'听':best.shanten}<small>${best.shanten===0?'牌':'向听'}</small></strong></div><div class="metric"><span>未见有效进张</span><strong>${best.ukeire}<small>张</small></strong></div></div><p class="coach-explanation">${explanation}</p></div><table class="candidate-table"><thead><tr><th>候选出牌</th><th>向听</th><th>进张</th></tr></thead><tbody>${list.slice(0,4).map((a,i)=>`<tr class="${i===0?'topchoice':''}"><td><span class="rankdot">0${i+1}</span>${tileName(a.tile)}${a.tile===selectedTile?' · 已选':''}</td><td>${a.shanten===0?'听牌':a.shanten}</td><td>${a.ukeire} 张</td></tr>`).join('')}</tbody></table>${actual?`<div class="selected-detail">${review?'你当时打出':'当前选择'} ${tileName(selectedTile)}：${shantenText(actual.shanten)}，${actual.ukeire} 张未见有效进张。${actual.shanten===best.shanten&&actual.ukeire===best.ukeire?'与首选牌的向听、进张相同。':actual.shanten>best.shanten?'比首选多 '+(actual.shanten-best.shanten)+' 向听。':'比首选少 '+(best.ukeire-actual.ukeire)+' 张进张。'}</div>`:''}<div class="info-footnote">${icon('info')}<span>未见牌可能在对手手中。此排序衡量牌效，不代表完整策略最优，也不包含放炮风险。</span></div></section>`;
}
function getReviewGame(){return reviewSeed===null||reviewSeed===game.seed?game:archive.find(a=>a.seed===reviewSeed)||game;}
function reviewPage(){
  const rg=getReviewGame(),decisions=rg.events.filter(e=>e.review),idx=Math.min(reviewIndex,Math.max(0,decisions.length-1)),event=decisions[idx];
  const options=[game,...archive.filter(a=>a.seed!==game.seed)].filter(g=>g.events.some(e=>e.review));
  let header=intro('LOOK BACK · LEARN FORWARD','让每一手，都留下答案。','回到出牌之前，用当时可见的信息重新比较。',btn(icon('upload')+'导入','import')+btn(icon('download')+'导出','export-review'));
  if(!event)return `${header}<section class="empty-state">${icon('review')}<h2>第一份牌谱，从这一局开始。</h2><p>完成一次出牌后，就能逐步回看。<br>也可以导入从听川导出的牌谱。</p>${btn('回到牌桌','live','primary')}<button type="button" class="import-button" data-action="import" style="margin-top:16px">导入已有牌谱</button></section>`;
  const before=structuredClone(event.state);before.players[0].hand=event.review.obs.hand;before.players[0].melds=event.review.obs.melds;before.players[0].discards.pop();before.last=null;before.phase='discard';before.turn=0;
  const good=decisions.filter(e=>reviewGrade(e.review).good).length;
  return `${header}<div class="game-layout"><div><div class="board-wrap"><div class="board-bar"><div class="board-bar-title"><span>第 ${idx+1} 次出牌</span><span class="badge">出牌前</span></div><select id="review-source" aria-label="选择复盘牌局" style="max-width:155px;border:0;background:transparent;font-size:10px;color:#7e9070">${options.map(g=>`<option value="${g.seed}" ${g.seed===rg.seed?'selected':''}>${g.seed===game.seed?'当前牌局':'历史牌局 '+String(g.seed).slice(-5)} · ${g.phase==='ended'?'已结束':'进行中'}</option>`).join('')}</select></div>${board(before,{review:true,highlight:event.review.chosen})}<div class="review-controls">${btn(icon('back'),'prev-review','small',idx===0)}<input id="review-range" type="range" min="0" max="${decisions.length-1}" value="${idx}" aria-label="复盘步骤"><span class="review-count">${idx+1} / ${decisions.length}</span>${btn(icon('forward'),'next-review','small',idx>=decisions.length-1)}</div><div class="review-text">你打出 <strong>${tileName(event.review.chosen)}</strong>。${reviewGrade(event.review).good?'与牌效首选的向听数、有效进张相同。':'可以对比右侧推荐，看看有没有更宽的进张。'}<br><span class="muted">本页不使用之后的摸牌或对手暗手评价这一步。</span></div></div><section class="review-list"><div class="review-list-title">你的出牌记录 <span class="muted" style="font-size:9px;margin-left:8px">点击任一步回看</span></div><div style="max-height:290px;overflow:auto">${decisions.map((e,i)=>{const grade=reviewGrade(e.review);return `<button type="button" class="review-row ${i===idx?'active':''}" data-review-index="${i}"><span class="review-number">${String(i+1).padStart(2,'0')}</span><span>打出 ${tileName(e.review.chosen)}</span><span class="muted">推荐 ${tileName(e.review.analysis[0].tile)}</span><span class="grade ${grade.good?'':'miss'}">${grade.label}</span></button>`;}).join('')}</div></section></div><aside class="right-column">${coachPanel(event.review)}<section class="panel secondary-panel"><div class="panel-title">这局的研习记录</div><div class="session-metrics"><div class="session-metric"><strong>${decisions.length}</strong><span>出牌决策</span></div><div class="session-metric"><strong>${good}</strong><span>符合牌效首选</span></div><div class="session-metric"><strong>${decisions.length-good}</strong><span>可比较的选择</span></div></div><div class="disclaimer">这些统计只反映牌效。保留安全牌、追求大牌等策略，有时会主动牺牲进张。</div></section></aside></div>`;
}
const puzzles=[
  {title:'多留一个搭子',subtitle:'入门 · 基础牌效',hand:[0,1,2,3,4,9,10,11,12,13,15,15,16,17],que:2,note:'观察孤张和相邻搭子。先比较打出后的向听数，再看有效进张。'},
  {title:'两面，还是边张？',subtitle:'入门 · 搭子取舍',hand:[0,1,2,3,4,5,9,10,11,12,13,15,17,17],que:2,note:'已有三组顺子时，比较剩余搭子的连接能力。'},
  {title:'七对的另一条路',subtitle:'进阶 · 牌型分岔',hand:[0,0,1,1,3,3,5,5,9,9,11,11,13,16],que:2,note:'川麻七对允许四张相同牌算两对。分析器同时比较普通牌型与七对向听。'},
  {title:'定缺之后的次序',subtitle:'入门 · 清缺门',hand:[0,1,3,6,9,10,11,12,13,14,18,19,20,20],que:0,note:'还有缺门牌时，只能从缺门牌里选择。比较清缺过程中对后续牌型的影响。'},
  {title:'一张孤牌的去留',subtitle:'进阶 · 有效进张',hand:[0,1,2,3,3,4,5,6,9,10,11,14,16,17],que:2,note:'连接方式多时，首选牌可能不止一张。相同向听与进张都算通过。'}
];
function puzzleObs(p){return {hand:p.hand,melds:[],que:p.que,seen:counts(p.hand),remaining:40};}
function trainPage(){
  const p=puzzles[trainIndex],list=analyze(puzzleObs(p)),best=list[0],chosen=trainChoice===null?null:list.find(a=>a.tile===p.hand[trainChoice]),good=chosen&&chosen.shanten===best.shanten&&chosen.ukeire===best.ukeire;
  return `${intro('SMALL PRACTICE · BETTER INSTINCT','把判断，练成直觉。','每天几手何切，从看懂进张，到形成自己的判断。')}<div class="train-layout"><section class="train-card"><div class="train-header"><div><h2>${p.title}</h2><p>${p.subtitle} · 第 ${trainIndex+1} / ${puzzles.length} 题</p></div><span class="tinybadge">牌效练习</span></div><div class="train-felt"><div class="train-prompt">这一手，你会打哪一张？</div><div class="hand">${p.hand.map((t,i)=>tile(t,{action:'train-tile-'+i,selected:trainChoice===i,forced:Math.floor(t/9)===p.que})).join('')}</div><div class="train-meta"><span>定缺：${SUITS[p.que]}</span><span>只考虑手牌，无额外公开弃牌</span></div></div><div class="train-bottom"><div class="train-result">${chosen?`${good?'✓ 这也是牌效推荐。':'再比较一下这个选择。'}<br>打 ${tileName(chosen.tile)}：${shantenText(chosen.shanten)}，${chosen.ukeire} 张未见有效进张。首选打 ${tileName(best.tile)}：${shantenText(best.shanten)}，${best.ukeire} 张。`:`<span class="muted">点击一张合法出牌，查看牌效比较。</span>`}</div><div class="disclaimer">${p.note}</div><div class="train-navigation">${btn(icon('back')+'上一题','prev-train','small',trainIndex===0)}<div class="train-dots">${puzzles.map((_,i)=>`<button type="button" class="train-dot ${i===trainIndex?'active':''}" data-action="train-${i}" aria-label="第${i+1}题"></button>`).join('')}</div>${btn('下一题'+icon('forward'),'next-train','small',trainIndex===puzzles.length-1)}</div></div></section>${chosen?analysisPanel(list,{analysis:list,chosen:chosen.tile}):`<aside class="panel"><div class="panel-title">这一题，怎么想？${icon('leaf')}</div><h2 class="coach-title">先想一手，再看答案。</h2><p class="coach-desc">${p.note}</p><div class="mini-steps"><div class="mini-step current"><b>1</b><span>确认哪些牌可以合法打出</span></div><div class="mini-step"><b>2</b><span>比较距离听牌还有多远</span></div><div class="mini-step"><b>3</b><span>比较未见有效进张</span></div></div></aside>`}</div>`;
}
function rulesPage(){
  return `${intro('THE RULES · OUR COMMON GROUND','先懂规则，再谈好选择。','当前规则版本：血战研习 v1。自摸加番、五番封顶，启用过手胡。')}<div class="rules-grid"><section class="rules-section rules-wide"><h2>这一桌，怎么打？</h2><p>新开一局时可选择普通血战或幺鸡万能牌，并独立选择换三张后定缺、或直接定缺。好友房由房主在大厅选择，开局后规则固定。</p><p>${variantHint} 幺鸡参与碰杠时显示实际用牌；用幺鸡碰牌后补杠，原幺鸡留在杠内。已有幺鸡杠后，再摸到该杠对应的牌，可点击“换回幺鸡”替换其中一张，不重复收杠分，也不额外补牌。</p><p>万能牌胡牌按合法组成中番数最高的牌型计分。幺鸡仅免于缺门限制，其他缺门牌未清完时仍不能碰、杠、胡。</p><div class="rules-highlight"><div><strong>血战到底</strong><span>三家胡牌或牌墙摸完结束</span></div><div><strong>换三张 · 定缺</strong><span>108 张万、筒、条，无字牌</span></div><div><strong>自摸加 1 番</strong><span>参与本次结算的未胡玩家支付</span></div><div><strong>五番封顶</strong><span>基础分 1，最高单份 32 分</span></div></div><p>启用换三张时，开局四家同时换出同一花色的三张牌（幺鸡按条牌交换），每局随机确定顺时针、逆时针或对家交换，然后各自定缺；不换三张时直接定缺。庄家先出牌。缺门未清时，只能打缺门牌，不能碰、杠、胡；清完缺门后才能正常碰杠胡。不能吃牌。</p><p>过手胡：放过可胡的点炮牌后，在自己再次摸牌前，同番或更低番的点炮牌都不能胡，即使是另一张听牌。仅实际结算番数更高时可胡；再次放过更高番，门槛随之提高。碰牌、换一张出牌或其他玩家摸牌不会解除；自己摸牌（包括杠后补牌）后解除，不限制自摸。</p><p>响应按胡牌优先处理，允许一炮多响；无人胡时按座次处理碰杠。胡牌者退出后续行牌与新的杠牌、自摸支付，其余玩家继续，直至三家胡牌或牌墙耗尽。</p></section><section class="rules-section"><h2>番型与计分</h2><table class="rules-table"><tbody>${[['平胡','0 番'],['碰碰胡','+1 番'],['清一色','+2 番'],['七对','+2 番'],['金钩钓（四副露）','+1 番'],['根（同种牌四张）','每根 +1 番'],['自摸 / 杠上花 / 抢杠胡 / 海底','分别 +1 番'],['无鸡（仅幺鸡玩法，含副露）','+1 番'],['单份支付','2 的番数次方，上限 32 分']].map(([a,b])=>`<tr><td>${a}</td><td>${b}</td></tr>`).join('')}</tbody></table><p>龙七对按“七对＋根”计算；同一个根只计一次。可以叠加的番型累计后封顶。点炮由放炮者支付；自摸由其他尚未胡牌者分别支付。本版抢杠胡只针对补杠，暂不启用呼叫转移、天地胡、将对、幺九等扩展番型。</p></section><section class="rules-section"><h2>杠牌与流局结算</h2><table class="rules-table"><tbody>${[['直杠','放杠者支付 2 分'],['暗杠','其他未胡玩家各付 2 分'],['补杠','其他未胡玩家各付 1 分'],['查花猪','未清缺门者向非花猪各付 32 分'],['查大叫','未听牌者向听牌者支付最大可胡分'],['未听牌退税','未听牌/花猪退回已收杠分']].map(([a,b])=>`<tr><td>${a}</td><td>${b}</td></tr>`).join('')}</tbody></table><p>仅牌墙耗尽且仍有多家未胡时查叫。查花猪包含向已胡牌者支付；查大叫只在尚未胡牌的非花猪中进行，不额外计自摸、杠上花或海底。听牌按规则允许组成胡牌的牌型判定，不因有效牌已全部可见而取消。无牌可补时不能杠。</p></section><section class="rules-section"><h2>如何理解牌效建议？</h2><ul><li>向听数：距离听牌还有多远；0 表示听牌。</li><li>未见有效进张：能降低向听数、且尚未被自己或公开信息占用的牌张上限。</li><li>先比较向听，再比较进张；并列时以手牌连接形状作稳定排序。</li><li>电脑与分析器只使用该座位的可见信息。</li><li>当前不评估放炮概率、对手暗手分布或长期预期收益。</li><li>复盘的“牌效推荐”允许多个向听、进张相同的选择，不能用于认定完整策略对错。</li></ul></section><section class="rules-section"><h2>这版可以做什么？</h2><ul><li>与三个基础电脑完成血战对局。</li><li>逐步回看自己的每次出牌和当时局面。</li><li>练习五道何切题，比较合法出牌的牌效。</li><li>在同一浏览器保留当前牌局和最近三份历史。</li><li>导出、导入听川自己的 JSON 牌谱。</li></ul><p>好友同桌已支持真人与电脑混合对战、断线重连和结算后复盘；两种麻将玩法与两种开局方式均可使用。</p><div style="display:flex;gap:9px;margin-top:16px">${btn(icon('download')+'导出牌谱','export','small')}${btn(icon('upload')+'导入牌谱','import','small')}${btn('牌面图鉴','tile-gallery','small')}</div></section></div>`;
}
function schedule(){
  clearTimeout(timer);if(page!=='live'||paused||game.phase==='ended'||game.phase==='swap'||game.phase==='que')return;
  if(game.phase==='respond'&&game.decision||game.phase==='discard'&&game.turn===0)return;
  timer=setTimeout(()=>{
    try{const did=aiStep(game);if(did){assertGame(game);save();render();checkEnd();}}
    catch(e){paused=true;render();toast('对局已暂停：'+e.message);console.error(e);}
  },speed);
}
function checkEnd(){if(game.phase==='ended'){storeArchive(game);if(settledSeed!==game.seed&&page==='live'){settledSeed=game.seed;settlement();}}}
function act(fn){try{clearTimeout(timer);fn();selected=[];playSound();assertGame(game);save();render();checkEnd();}catch(e){toast(e.message);schedule();}}
function changePage(id){clearTimeout(timer);page=id;selected=[];if(id==='review')reviewIndex=Math.min(reviewIndex,Math.max(0,getReviewGame().events.filter(e=>e.review).length-1));render();window.scrollTo({top:0,behavior:'instant'});}
function openModal(title,body){clearTimeout(timer);$('#modal-root').innerHTML=`<div class="modal-backdrop"><section class="modal" role="dialog" aria-modal="true" aria-label="${esc(title)}"><div class="modal-head"><h2>${title}</h2><button type="button" class="iconbutton" data-action="close-modal" aria-label="关闭弹窗">${icon('close')}</button></div><div class="modal-body">${body}</div></section></div>`;setTimeout(()=>$('#modal-root button')?.focus(),0);}
function closeModal(){ $('#modal-root').innerHTML='';schedule();}
function settlement(){
  const players=[...game.players].sort((a,b)=>b.score-a.score);
  openModal('这一局，落定了。',`<p>${game.wall.length?'三家已胡，血战结束。':'牌墙摸完，查花猪、查大叫与退税已结算。'}牌谱已自动保存在本机。</p>${players.map((p,i)=>`<div class="settle-row"><span class="review-number">0${i+1}</span><span class="avatar">${p.name[0]}</span><span>${p.name}<br><small>${p.win?p.win.tags.map(t=>t.name).join(' · ')+' / '+p.win.fan+' 番':!clearQue(p,game.rules)?'未清缺门':waits(p,game.rules).length?'已听牌':'未听牌'}</small></span><strong class="${p.score>=0?'positive':'negative'}">${p.score>0?'+':''}${p.score}</strong></div>`).join('')}<div class="ledger">${game.ledger.length?game.ledger.map(l=>`<div><span>${game.players[l.from].name} → ${game.players[l.to].name} · ${l.reason}</span><span>${l.amount} 分</span></div>`).join(''):'本局没有分数转移。'}</div><div class="modal-actions">${btn('再开一局','confirm-new')}${btn('逐步复盘','modal-review','primary')}</div>`);
}
function exportGame(g){const payload={format:'tingchuan-replay',version:VERSION,exportedAt:new Date().toISOString(),rules:normalizeRules(g.rules),game:g};const blob=new Blob([JSON.stringify(payload)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`听川-牌谱-${g.seed}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('牌谱已导出，可以在复盘页重新导入。');}
function importModal(){openModal('带回一份牌谱。',`<p>支持听川 v0.1.0 至 v${VERSION} 导出的 JSON 牌谱。导入后保留当前对局，在复盘页打开。</p><input type="file" id="replay-file" accept=".json,application/json" aria-label="选择牌谱文件"><div class="disclaimer">牌谱包含本地完整对局状态，仅用于离线复盘。当前没有多人游戏与在线分享服务。</div>`);}
function newGame(){openModal('选择本局玩法',`<p>选择麻将玩法和开局方式。当前牌局会保留在历史中，方便复盘。</p>${variantPicker(normalizeRules(game.rules))}<div class="modal-actions">${btn('返回当前局','close-modal')}${btn('开始新局','confirm-new','primary')}</div>`);}
function startNew(){storeArchive(game);game=createGame(undefined,null,$('#variant-joker')?{joker:$('#variant-joker').value==='true',swap:$('#variant-swap').value==='true'}:game.rules);selected=[];paused=false;settledSeed=null;reviewSeed=null;reviewIndex=0;analysisCache={key:'',data:[]};closeModal();save();changePage('live');}

document.addEventListener('click',e=>{
  const nav=e.target.closest('[data-page]');if(nav){changePage(nav.dataset.page);return;}
  const ri=e.target.closest('[data-review-index]');if(ri){reviewIndex=Number(ri.dataset.reviewIndex);render();return;}
  const ht=e.target.closest('[data-tile-index]');if(ht){
    const i=Number(ht.dataset.tileIndex);if(paused)return;
    if(game.phase==='swap'){if(selected.includes(i))selected=selected.filter(x=>x!==i);else if(selected.length<3)selected.push(i);else return toast('先取消一张选择，再换另一张。');}
    else if(game.phase==='discard'&&game.turn===0){if(!legalDiscards(game).includes(game.players[0].hand[i]))return toast('缺门还没清完，请先打缺'+SUITS[game.players[0].que]+'的牌。');selected=selected[0]===i?[]:[i];}
    render();return;
  }
  const action=e.target.closest('[data-action]')?.dataset.action;if(!action)return;
  if(action==='auto-swap'){selected=recommendSwap(game.players[0].hand,game.rules);render();}
  else if(action==='swap')act(()=>swapTiles(game,selected));
  else if(action.startsWith('que-'))act(()=>setQue(game,Number(action.split('-')[1])));
  else if(action==='discard'){if(selected.length===1)act(()=>discard(game,game.players[0].hand[selected[0]]));}
  else if(action==='pass')act(()=>resolveResponse(game,'pass'));
  else if(action.startsWith('claim-'))act(()=>resolveResponse(game,action.slice(6)));
  else if(action.startsWith('self-')){const [,type,t,j]=action.split('-');act(()=>selfAction(game,type,t===''?undefined:Number(t),0,Number(j||0)));}
  else if(action==='tile-gallery')tileGallery();
  else if(action==='new')newGame();else if(action==='confirm-new')startNew();
  else if(action==='pause'){paused=!paused;render();}
  else if(action==='sound'){sound=!sound;playSound();render();toast(sound?'轻音效已开启':'音效已关闭');}
  else if(action==='export')exportGame(game);else if(action==='export-review')exportGame(getReviewGame());
  else if(action==='import')importModal();
  else if(action==='settlement')settlement();
  else if(action==='review-current'||action==='modal-review'){closeModal();reviewSeed=null;reviewIndex=0;changePage('review');}
  else if(action==='live')changePage('live');
  else if(action==='close-modal')closeModal();
  else if(action==='next-review'){reviewIndex++;render();}else if(action==='prev-review'){reviewIndex=Math.max(0,reviewIndex-1);render();}
  else if(action==='next-train'){trainIndex=Math.min(puzzles.length-1,trainIndex+1);trainChoice=null;render();}
  else if(action==='prev-train'){trainIndex=Math.max(0,trainIndex-1);trainChoice=null;render();}
  else if(action.startsWith('train-tile-')){const i=Number(action.slice(11)),p=puzzles[trainIndex];if(!analyze(puzzleObs(p)).some(a=>a.tile===p.hand[i]))return toast('请先打定缺花色：'+SUITS[p.que]);trainChoice=i;playSound();render();}
  else if(action.startsWith('train-')){trainIndex=Number(action.slice(6));trainChoice=null;render();}
  else if(action==='help')openModal('慢慢打，慢慢懂。',`<p>① 换三张：点击同一花色的三张牌，再确认。<br>② 定缺：选择本局优先清完的花色。<br>③ 出牌：先点击手牌，再点“打出”或按 Enter。<br>④ 响应：出现可碰、杠、胡的牌时，选择操作或过。<br>⑤ 复盘：每次出牌都会记录，可随时到逐步复盘回看。</p><div class="disclaimer">当前是本地人机研习版本。牌效建议不包含防守风险与长期收益，电脑为基础牌效策略。对局自动保存在此浏览器，清理浏览器数据前请导出牌谱。</div><div class="modal-actions">${btn('开始研习','close-modal','primary')}</div>`);
});
document.addEventListener('change',async e=>{
  if(e.target.id==='variant-joker')$('.variant-hint').textContent=e.target.value==='true'?variantHint:'血战到底，自摸加番，五番封顶。两种开局方式都保留定缺。';
  if(e.target.id==='speed'){speed=Number(e.target.value);schedule();}
  if(e.target.id==='review-source'){reviewSeed=Number(e.target.value);reviewIndex=0;render();}
  if(e.target.id==='replay-file'&&e.target.files.length){
    try{
      const file=e.target.files[0];if(file.size>12*1024*1024)throw Error('文件过大，请选择 12MB 以内的听川牌谱');
      const parsed=JSON.parse(await file.text()),g=parsed.game;
      if(parsed.format!=='tingchuan-replay'||!isCompatibleVersion(parsed.version)||g?.ruleVersion!==RULES.version||!Array.isArray(g.events)||g.events.length>1000||!Number.isInteger(g.seed)||g.players?.length!==4)throw Error('不是兼容的听川牌谱，请使用兼容版本导出的文件');
      validateImportedGame(g);restoreGame(g);archive=[g,...archive.filter(a=>a.seed!==g.seed)].slice(0,3);reviewSeed=g.seed;reviewIndex=0;save();closeModal();changePage('review');toast('牌谱已导入。');
    }catch(err){toast('导入失败：'+err.message);}
  }
});
function validateImportedGame(g){
  const validTile=t=>Number.isInteger(t)&&t>=0&&t<27;
  const checkState=s=>{if(!s||!Array.isArray(s.wall)||!s.wall.every(validTile)||s.players?.length!==4)throw Error('牌局状态不完整');for(const p of s.players){if(!Array.isArray(p.hand)||p.hand.length>14||!p.hand.every(validTile)||!Array.isArray(p.discards)||p.discards.length>108||!p.discards.every(validTile)||!Array.isArray(p.melds)||p.melds.length>4||!p.melds.every(m=>validTile(m.tile)&&['pong','kong','ankong','addkong'].includes(m.type)&&(!m.tiles||Array.isArray(m.tiles)&&m.tiles.length===(m.type==='pong'?3:4)&&m.tiles.every(t=>validTile(t)&&(t===m.tile||s.rules?.joker&&t===18))))||!Number.isFinite(p.score)||![-1,0,1,2].includes(p.que)||typeof p.name!=='string'||p.name.length>20)throw Error('牌张或玩家数据不合法');}assertGame(s);};
  checkState(g);
  // Recompute imported recommendations rather than trusting annotations in a file.
  for(const e of g.events){checkState(e.state);if(e.review){const o=e.review.obs;if(!o||!Array.isArray(o.hand)||!o.hand.every(validTile)||o.hand.length>14||!Array.isArray(o.seen)||o.seen.length!==27||!o.seen.every(n=>Number.isInteger(n)&&n>=0&&n<=4)||![-1,0,1,2].includes(o.que)||!Array.isArray(o.melds)||!validTile(e.review.chosen))throw Error('分析记录不合法');e.review.analysis=analyze(o);if(!e.review.analysis.some(a=>a.tile===e.review.chosen))throw Error('牌谱含非法出牌');}}
  // Names from external JSON never enter markup as raw HTML.
  g.players.forEach(p=>p.name=esc(p.name));for(const e of g.events)e.state.players.forEach(p=>p.name=esc(p.name));
}
document.addEventListener('input',e=>{if(e.target.id==='review-range'){reviewIndex=Number(e.target.value);render();}});
document.addEventListener('keydown',e=>{
  if(e.key==='Escape'){if($('#modal-root').children.length)closeModal();else{selected=[];render();}}
  if(e.key==='Enter'&&page==='live'&&!$('#modal-root').children.length&&!paused&&selected.length===1&&game.phase==='discard'&&game.turn===0){e.preventDefault();act(()=>discard(game,game.players[0].hand[selected[0]]));}
  if(e.key==='Tab'&&$('#modal-root').children.length){const focusable=[...$('#modal-root').querySelectorAll('button,input,select,a')].filter(el=>!el.disabled);const first=focusable[0],last=focusable.at(-1);if(e.shiftKey&&document.activeElement===first){e.preventDefault();last.focus();}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first.focus();}}
});
document.addEventListener('visibilitychange',()=>{if(document.hidden)clearTimeout(timer);else schedule();});
window.addEventListener('beforeunload',()=>save());
render();save();

function tileGallery(){
  openModal('万、筒、条，一张不差。', '<p>本桌使用中文麻将牌面，三门各九种，每种四张，共 108 张。</p>'+[0,1,2].map(s=>'<div class="tile-gallery-row"><div class="tile-gallery-label">'+SUITS[s]+'子</div><div class="tile-gallery-grid">'+Array.from({length:9},(_,n)=>'<div>'+tile(s*9+n)+'<small>'+tileName(s*9+n)+'</small></div>').join('')+'</div></div>').join('')+'<p class="asset-credit">牌面：<a href="https://github.com/lietxia/mahjong_graphic" target="_blank" rel="noopener noreferrer">lietxia / mahjong_graphic</a> · M+ 字型授权条款。保留原始牌面排列与颜色。</p>');
}
