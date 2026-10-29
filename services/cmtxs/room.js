import {VERSION,RULES,normalizeRules,createGame,recommendSwap,recommendQue,exchangeTiles,setAllQue,legalDiscards,selfActions,chooseSelfAction,selfAction,chooseDiscard,discard,responseActions,chooseResponse,resolveResponses,analyze,observe,assertGame} from '../../source/cmtxs/engine.js';
import {DEFAULT_BOT_STYLES,isBotStyle,normalizeBotStyles} from '../../source/cmtxs/bot-policy.js';
import {EMOTE_COOLDOWN_MS,EMOTE_DURATION_MS,isEmoteId} from '../../source/cmtxs/emotes.js';
import {newMatch,matchRounds,matchComplete,recordRound,matchView} from './match.js';

export const PROTOCOL=1;
const copy=x=>structuredClone(x);
const secure=()=>crypto.getRandomValues(new Uint32Array(1))[0]/4294967296;
export const secret=()=>Array.from(crypto.getRandomValues(new Uint8Array(32)),n=>n.toString(16).padStart(2,'0')).join('');
export const roomCode=()=>Array.from(crypto.getRandomValues(new Uint8Array(8)),n=>'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'[n%32]).join('');
export const cleanName=n=>String(n||'牌友').trim().replace(/[\x00-\x1f\x7f]/g,'').slice(0,12)||'牌友';
const fail=m=>{throw Error(m);};
export class Room {
  constructor(data=null,{now=Date.now,random=secure,timeout=30000,botDelay=700}={}){
    this.now=now;this.random=random;this.timeout=timeout;this.botDelay=botDelay;
    this.d=data?copy(data):{protocol:PROTOCOL,code:'',created:now(),updated:now(),expires:now()+86400000,revision:0,decision:0,members:Array(4).fill(null),game:null,submissions:{},deadline:null,botAt:null,reviews:[],recent:[]};
    this.d.admissions??=[];
    this.d.emoteLimits??=Array.from({length:4},()=>({at:null,ids:[]}));
    this.d.rules=normalizeRules(this.d.rules);
    this.d.roundId??=this.d.game?1:0;
    this.d.match??=newMatch();this.d.matchArchive??=[];
    this.d.botStyles=normalizeBotStyles(this.d.botStyles??this.d.game?.botStyles,data?Array(4).fill('balanced'):DEFAULT_BOT_STYLES);
    if(this.d.game){this.d.match.started=true;recordRound(this.d.match,this.d.game,this.d.roundId);}
  }
  admission(requestId){
    if(requestId==null)return null;
    if(typeof requestId!=='string'||!/^[a-f0-9]{64}$/.test(requestId))fail('入座请求无效，请刷新后重试');
    const previous=this.d.admissions.find(a=>a.id===requestId);
    return previous?this.credential(previous.seat):null;
  }
  credential(seat){return {code:this.d.code,token:this.d.members[seat].token,seat,protocol:PROTOCOL};}
  init(code,name,requestId=null){
    this.check();const previous=this.admission(requestId);
    if(this.d.code){if(previous&&previous.seat===0&&code===this.d.code)return previous;fail('房间号已存在，请重新创建');}
    this.d.code=code;return this.join(name,requestId);
  }
  serialize(){return copy(this.d);}
  expired(){return this.now()>this.d.expires;}
  check(){if(this.expired())fail('房间已过期，请重新创建');}
  bump(){this.d.revision++;this.d.updated=this.now();}
  seat(token){if(typeof token!=='string'||token.length!==64)return -1;return this.d.members.findIndex(m=>m?.token===token);}
  join(name,requestId=null){
    this.check();const previous=this.admission(requestId);if(previous)return previous;
    if(this.d.game)fail('对局已开始，只能由原玩家重连');
    const seat=this.d.members.findIndex(m=>!m);if(seat<0)fail('房间已满');
    const token=secret();this.d.members[seat]={name:cleanName(name),token,ready:false,online:false,auto:false,commands:[]};
    this.d.botStyles[seat]='balanced';
    if(requestId)this.d.admissions.push({id:requestId,seat});this.bump();return this.credential(seat);
  }
  presence(seat,online){const m=this.d.members[seat];if(!m)return;if(m.online!==online){m.online=online;this.bump();}}
  pending(seat){
    const g=this.d.game;if(!g||g.phase==='ended')return false;
    if(['swap','que'].includes(g.phase))return !Object.hasOwn(this.d.submissions,seat);
    if(g.phase==='discard')return g.turn===seat;
    return responseActions(g,seat).length>0&&!Object.hasOwn(this.d.submissions,seat);
  }
  newDecision(){
    recordRound(this.d.match,this.d.game,this.d.roundId);
    this.d.decision++;this.d.submissions={};this.d.deadline=this.now()+this.timeout;this.d.botAt=null;
    const g=this.d.game;if(!g||g.phase==='ended'){this.d.deadline=null;return;}
    for(let s=0;s<4;s++)if(!this.d.members[s]||this.d.members[s].auto){
      if(g.phase==='swap')this.d.submissions[s]=recommendSwap(g.players[s].hand,g.rules);
      if(g.phase==='que')this.d.submissions[s]=recommendQue(g.players[s].hand,g.rules);
      if(g.phase==='respond')this.d.submissions[s]=chooseResponse(g,s);
    }
    if(g.phase==='discard'&&(!this.d.members[g.turn]||this.d.members[g.turn].auto))this.d.botAt=this.now()+this.botDelay;
  }
  collectLogs(){
    const g=this.d.game;if(!g)return;
    this.d.recent.push(...g.events.filter(e=>!['draw','swap','que'].includes(e.type)).map(e=>({type:e.type,text:e.text})));
    this.d.recent=this.d.recent.slice(-12);g.events=[];assertGame(g);recordRound(this.d.match,g,this.d.roundId);
  }
  advance(){
    // Automatic transitions are bounded; discard decisions always get their own timer.
    for(let i=0;i<8;i++){
      const g=this.d.game;if(!g||g.phase==='ended')break;
      const pending=[0,1,2,3].some(s=>this.pending(s));
      if(g.phase==='discard'||pending)break;
      if(g.phase==='swap')exchangeTiles(g,[0,1,2,3].map(s=>this.d.submissions[s]));
      else if(g.phase==='que')setAllQue(g,[0,1,2,3].map(s=>this.d.submissions[s]));
      else if(g.phase==='respond')resolveResponses(g,[0,1,2,3].map(s=>this.d.submissions[s]??'pass'));
      this.collectLogs();this.newDecision();
    }
  }
  playDiscard(seat,tile){
    const g=this.d.game;if(!legalDiscards(g,seat).includes(tile))fail(g.rules?.joker&&tile===18?'幺鸡是万能牌，不能打出':'请先打完定缺花色');
    const obs=observe(g,seat);discard(g,tile,seat);this.d.reviews.push({seat,obs,chosen:tile});
  }
  automate(seat){
    const g=this.d.game;
    if(g.phase==='swap')this.d.submissions[seat]=recommendSwap(g.players[seat].hand,g.rules);
    else if(g.phase==='que')this.d.submissions[seat]=recommendQue(g.players[seat].hand,g.rules);
    else if(g.phase==='respond')this.d.submissions[seat]=chooseResponse(g,seat);
    else if(g.phase==='discard'){
      const a=chooseSelfAction(g,seat);
      if(a)selfAction(g,a.type,a.tile,seat,a.jokers??0);else this.playDiscard(seat,chooseDiscard(g,seat));
      this.collectLogs();this.newDecision();
    }
  }
  sendEmote(seat,c){
    this.check();if(!this.d.members[seat])fail('请先加入房间');
    if(!c||typeof c.id!=='string'||!/^[\w-]{1,80}$/.test(c.id)||!isEmoteId(c.sticker))fail('表情无效，请刷新网页');
    const limit=this.d.emoteLimits[seat];
    if(limit.ids.includes(c.id))return {id:c.id,duplicate:true};
    if(!this.d.game)fail('开局后才能发送牌桌表情');
    const at=this.now();if(limit.at!=null&&at-limit.at<EMOTE_COOLDOWN_MS)fail('表情发送太快，请稍候再试');
    limit.at=at;limit.ids.push(c.id);limit.ids=limit.ids.slice(-16);
    // Only the abuse-control cursor is saved. The image event itself is live-only.
    return {id:c.id,event:{type:'emote',id:c.id,seat,sticker:c.sticker,at,expiresAt:at+EMOTE_DURATION_MS}};
  }
  command(seat,c){
    this.check();const m=this.d.members[seat];if(!m)fail('请先加入房间');
    if(!c||typeof c.id!=='string'||!/^[\w-]{1,80}$/.test(c.id))fail('操作编号无效');
    if(m.commands.includes(c.id))return {id:c.id,duplicate:true};
    const g=this.d.game;
    if(c.action==='bot-config'){
      if(seat!==0)fail('只有房主可以选择电脑性格');
      if(g&&g.phase!=='ended')fail('本局已开始，电脑性格已锁定');
      if(this.d.match.rounds&&this.d.match.started&&!matchComplete(this.d.match))fail('比赛进行中，电脑性格已锁定');
      if(!Array.isArray(c.botStyles)||c.botStyles.length!==4||!c.botStyles.every(isBotStyle))fail('电脑性格设置无效');
      if(this.d.members.some((p,s)=>p&&c.botStyles[s]!=='balanced'))fail('真人托管固定使用均衡型');
      this.d.botStyles=[...c.botStyles];this.d.members.forEach(p=>{if(p)p.ready=false;});
    }else if(c.action==='match-config'){
      if(seat!==0)fail('只有房主可以修改赛制');
      if(this.d.match.started||g)fail('本场已经开始，赛制已锁定');
      if(!matchRounds(c.rounds))fail('赛制设置无效');
      this.d.match=newMatch(c.rounds);this.d.members.forEach(p=>{if(p)p.ready=false;});
    }else if(c.action==='new-match'){
      if(seat!==0)fail('只有房主可以新开一场');
      if(!g||g.phase!=='ended')fail('请先完成当前对局');
      if(this.d.match.rounds&&!matchComplete(this.d.match))fail('请先打完本场比赛');
      if(!matchRounds(c.rounds))fail('赛制设置无效');
      this.d.matchArchive=[matchView(this.d.match),...this.d.matchArchive].slice(0,3);
      this.d.match=newMatch(c.rounds);this.d.game=null;this.d.submissions={};this.d.deadline=null;this.d.botAt=null;this.d.reviews=[];this.d.recent=[];
      this.d.members.forEach(p=>{if(p){p.ready=false;p.auto=false;}});this.d.decision++;
    }else if(c.action==='configure'){
      if(seat!==0)fail('只有房主可以修改玩法');if(g&&g.phase!=='ended')fail('本局已开始，不能更改玩法');
      if(this.d.match.rounds&&this.d.match.started&&!matchComplete(this.d.match))fail('比赛进行中，玩法已锁定');
      if(!c.rules||typeof c.rules.swap!=='boolean'||typeof c.rules.joker!=='boolean'||Object.keys(c.rules).some(k=>!['swap','joker'].includes(k)))fail('玩法设置无效');
      this.d.rules=normalizeRules(c.rules);this.d.members.forEach(p=>{if(p)p.ready=false;});
    }else if(c.action==='ready'){
      if(g&&g.phase!=='ended')fail('本局已开始');m.ready=!m.ready;
    }else if(c.action==='start'){
      if(seat!==0)fail('只有房主可以开局');if(g&&g.phase!=='ended')fail('本局已开始');
      if(this.d.members.some(p=>p&&!p.ready))fail('请等待所有真人准备');
      if(matchComplete(this.d.match))fail('本场已结束，请先新开一场');
      this.d.roundId++;this.d.match.started=true;
      this.d.game=createGame(0,this.random,{...this.d.rules,botStyles:this.d.botStyles.map((id,s)=>this.d.members[s]?'balanced':id)});this.d.game.reviewSeats=[];this.d.reviews=[];this.d.recent=[];
      this.d.members.forEach((p,s)=>{if(p){p.auto=false;p.ready=false;}this.d.game.players[s].name=p?.name||['竹隐','云岫','松风','清和'][s];});
      this.newDecision();this.advance();
    }else if(c.action==='auto'){
      if(!g||g.phase==='ended')fail('当前没有进行中的对局');
      m.auto=!!c.value;
      if(m.auto&&this.pending(seat)){
        if(g.phase==='discard')this.d.botAt=this.now()+this.botDelay;
        else{this.automate(seat);this.advance();}
      }else if(!m.auto&&g.phase==='discard'&&g.turn===seat)this.d.botAt=null;
    }else{
      if(!g||g.phase==='ended')fail('当前没有待操作的牌');
      if(c.decision!==this.d.decision)fail('牌局已更新，请按最新局面操作');
      if(m.auto)fail('请先取消托管');if(!this.pending(seat))fail('已提交或尚未轮到你');
      if(c.action==='swap'&&g.phase==='swap'){
        // Reuse the engine's atomic validation on a temporary game.
        const chosen=g.players.map((p,s)=>s===seat?c.indices:recommendSwap(p.hand,g.rules));exchangeTiles(copy(g),chosen);
        this.d.submissions[seat]=copy(c.indices);this.advance();
      }else if(c.action==='que'&&g.phase==='que'){
        if(![0,1,2].includes(c.suit))fail('定缺无效');this.d.submissions[seat]=c.suit;this.advance();
      }else if(c.action==='respond'&&g.phase==='respond'){
        if(c.choice!=='pass'&&!responseActions(g,seat).includes(c.choice))fail('当前不能进行此操作');
        this.d.submissions[seat]=c.choice;this.advance();
      }else if(c.action==='discard'&&g.phase==='discard'){
        this.playDiscard(seat,c.tile);this.collectLogs();this.newDecision();this.advance();
      }else if(c.action==='self'&&g.phase==='discard'){
        selfAction(g,c.choice,c.tile,seat,c.jokers??0);this.collectLogs();this.newDecision();this.advance();
      }else fail('操作与当前阶段不符');
    }
    m.commands.push(c.id);m.commands=m.commands.slice(-128);this.bump();return {id:c.id};
  }
  tick(){
    if(this.expired())return false;
    if(!this.d.game||this.d.game.phase==='ended')return false;
    if(this.d.botAt!=null&&this.now()>=this.d.botAt){this.automate(this.d.game.turn);this.advance();this.bump();return true;}
    if(this.d.deadline!=null&&this.now()>=this.d.deadline){
      const seats=[0,1,2,3].filter(s=>this.pending(s));
      for(const s of seats){if(this.d.members[s])this.d.members[s].auto=true;this.automate(s);}
      this.advance();this.bump();return true;
    }return false;
  }
  nextAlarm(){return Math.min(...[this.d.expires,this.d.deadline,this.d.botAt].filter(x=>x!=null));}
  view(seat){
    const d=this.d,g=d.game,m=d.members[seat];
    const view={type:'state',protocol:PROTOCOL,version:VERSION,features:['emotes-v1'],code:d.code,revision:d.revision,decision:d.decision,serverTime:this.now(),expires:d.expires,seat,host:0,rules:normalizeRules(g&&g.phase!=='ended'?g.rules:d.rules),deadline:d.deadline,phase:g?.phase||'lobby',pending:this.pending(seat),submitted:Object.hasOwn(d.submissions,seat),
      botStyles:[...d.botStyles],members:d.members.map((p,s)=>p?{name:p.name,ready:p.ready,online:p.online,auto:p.auto,bot:false}:{name:['竹隐','云岫','松风','清和'][s],style:d.botStyles[s],ready:true,online:true,auto:true,bot:true}),match:matchView(d.match),matchArchive:copy(d.matchArchive)};
    if(!g)return view;
    view.game={phase:g.phase,turn:g.turn,rules:normalizeRules(g.rules),botStyles:normalizeBotStyles(g.botStyles,Array(4).fill('balanced')),remaining:g.wall.length,last:copy(g.last),swapDirection:g.swapDirection,winners:[...g.winners],
      players:g.players.map((p,s)=>({seat:s,name:p.name,handCount:p.hand.length,hand:s===seat||p.won||g.phase==='ended'?[...p.hand]:null,melds:copy(p.melds),discards:[...p.discards],que:g.phase==='que'?-1:p.que,won:p.won,score:p.score,win:p.win?copy(p.win):null,passedHuFan:s===seat?p.passedHuFan:undefined})),ledger:copy(g.ledger),recent:copy(d.recent)};
    view.options={discards:g.phase==='discard'&&g.turn===seat?legalDiscards(g,seat):[],self:selfActions(g,seat),responses:responseActions(g,seat)};
    // The hand history contains only this seat's observations, released after settlement.
    if(g.phase==='ended')view.review=copy(d.reviews.filter(r=>r.seat===seat));
    view.auto=!!m?.auto;return view;
  }
}
