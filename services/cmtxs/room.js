import {VERSION,RULES,createGame,recommendSwap,recommendQue,exchangeTiles,setAllQue,legalDiscards,selfActions,selfAction,discard,responseActions,chooseResponse,resolveResponses,analyze,observe,assertGame} from '../../source/cmtxs/engine.js';

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
    this.d.decision++;this.d.submissions={};this.d.deadline=this.now()+this.timeout;this.d.botAt=null;
    const g=this.d.game;if(!g||g.phase==='ended'){this.d.deadline=null;return;}
    for(let s=0;s<4;s++)if(!this.d.members[s]||this.d.members[s].auto){
      if(g.phase==='swap')this.d.submissions[s]=recommendSwap(g.players[s].hand);
      if(g.phase==='que')this.d.submissions[s]=recommendQue(g.players[s].hand);
      if(g.phase==='respond')this.d.submissions[s]=chooseResponse(g,s);
    }
    if(g.phase==='discard'&&(!this.d.members[g.turn]||this.d.members[g.turn].auto))this.d.botAt=this.now()+this.botDelay;
  }
  collectLogs(){
    const g=this.d.game;if(!g)return;
    this.d.recent.push(...g.events.filter(e=>!['draw','swap','que'].includes(e.type)).map(e=>({type:e.type,text:e.text})));
    this.d.recent=this.d.recent.slice(-12);g.events=[];assertGame(g);
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
    const g=this.d.game;if(!legalDiscards(g,seat).includes(tile))fail('请先打完定缺花色');
    const obs=observe(g,seat);discard(g,tile,seat);this.d.reviews.push({seat,obs,chosen:tile});
  }
  automate(seat){
    const g=this.d.game;
    if(g.phase==='swap')this.d.submissions[seat]=recommendSwap(g.players[seat].hand);
    else if(g.phase==='que')this.d.submissions[seat]=recommendQue(g.players[seat].hand);
    else if(g.phase==='respond')this.d.submissions[seat]=chooseResponse(g,seat);
    else if(g.phase==='discard'){
      const actions=selfActions(g,seat),a=actions.find(x=>x.type==='hu')||actions[0];
      if(a)selfAction(g,a.type,a.tile,seat);else this.playDiscard(seat,analyze(observe(g,seat))[0].tile);
      this.collectLogs();this.newDecision();
    }
  }
  command(seat,c){
    this.check();const m=this.d.members[seat];if(!m)fail('请先加入房间');
    if(!c||typeof c.id!=='string'||!/^[\w-]{1,80}$/.test(c.id))fail('操作编号无效');
    if(m.commands.includes(c.id))return {id:c.id,duplicate:true};
    const g=this.d.game;
    if(c.action==='ready'){
      if(g&&g.phase!=='ended')fail('本局已开始');m.ready=!m.ready;
    }else if(c.action==='start'){
      if(seat!==0)fail('只有房主可以开局');if(g&&g.phase!=='ended')fail('本局已开始');
      if(this.d.members.some(p=>p&&!p.ready))fail('请等待所有真人准备');
      this.d.game=createGame(0,this.random);this.d.game.reviewSeats=[];this.d.reviews=[];this.d.recent=[];
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
        const chosen=g.players.map((p,s)=>s===seat?c.indices:recommendSwap(p.hand));exchangeTiles(copy(g),chosen);
        this.d.submissions[seat]=copy(c.indices);this.advance();
      }else if(c.action==='que'&&g.phase==='que'){
        if(![0,1,2].includes(c.suit))fail('定缺无效');this.d.submissions[seat]=c.suit;this.advance();
      }else if(c.action==='respond'&&g.phase==='respond'){
        if(c.choice!=='pass'&&!responseActions(g,seat).includes(c.choice))fail('当前不能进行此操作');
        this.d.submissions[seat]=c.choice;this.advance();
      }else if(c.action==='discard'&&g.phase==='discard'){
        this.playDiscard(seat,c.tile);this.collectLogs();this.newDecision();this.advance();
      }else if(c.action==='self'&&g.phase==='discard'){
        selfAction(g,c.choice,c.tile,seat);this.collectLogs();this.newDecision();this.advance();
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
    const view={type:'state',protocol:PROTOCOL,version:VERSION,code:d.code,revision:d.revision,decision:d.decision,serverTime:this.now(),expires:d.expires,seat,host:0,rules:RULES,deadline:d.deadline,phase:g?.phase||'lobby',pending:this.pending(seat),submitted:Object.hasOwn(d.submissions,seat),
      members:d.members.map((p,s)=>p?{name:p.name,ready:p.ready,online:p.online,auto:p.auto,bot:false}:{name:['竹隐','云岫','松风','清和'][s],ready:true,online:true,auto:true,bot:true})};
    if(!g)return view;
    view.game={phase:g.phase,turn:g.turn,remaining:g.wall.length,last:copy(g.last),swapDirection:g.swapDirection,winners:[...g.winners],
      players:g.players.map((p,s)=>({seat:s,name:p.name,handCount:p.hand.length,hand:s===seat||p.won||g.phase==='ended'?[...p.hand]:null,melds:copy(p.melds),discards:[...p.discards],que:g.phase==='que'?-1:p.que,won:p.won,score:p.score,win:p.win?copy(p.win):null,passedHuFan:s===seat?p.passedHuFan:undefined})),ledger:copy(g.ledger),recent:copy(d.recent)};
    view.options={discards:g.phase==='discard'&&g.turn===seat?legalDiscards(g,seat):[],self:selfActions(g,seat),responses:responseActions(g,seat)};
    // The hand history contains only this seat's observations, released after settlement.
    if(g.phase==='ended')view.review=copy(d.reviews.filter(r=>r.seat===seat));
    view.auto=!!m?.auto;return view;
  }
}
