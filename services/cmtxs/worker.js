import {DurableObject} from 'cloudflare:workers';
import {Room,roomCode} from './room.js';

const json=(body,status=200)=>Response.json(body,{status});
const allowed=(request,env)=>{const o=request.headers.get('Origin');return !o||(env.ALLOWED_ORIGINS||'').split(',').includes(o);};
function cors(response,request){
  const headers=new Headers(response.headers);headers.set('Access-Control-Allow-Origin',request.headers.get('Origin')||'*');headers.set('Vary','Origin');headers.set('Access-Control-Allow-Headers','Content-Type');headers.set('Access-Control-Allow-Methods','GET,POST,OPTIONS');headers.set('Cache-Control','no-store');
  return new Response(response.body,{status:response.status,headers});
}
export default {
  async fetch(request,env){
    if(!allowed(request,env))return json({error:'来源不允许'},403);
    if(request.method==='OPTIONS')return cors(new Response(null,{status:204}),request);
    const u=new URL(request.url);if(u.pathname==='/health')return cors(json({ok:true,protocol:1,version:'0.3.0',features:['match-v1']}),request);
    let response;
    try{
      if(u.pathname==='/rooms'&&request.method==='POST'){
        if(Number(request.headers.get('Content-Length'))>2048)return cors(json({error:'请求过大'},413),request);
        const raw=await request.text();if(raw.length>2048)return cors(json({error:'请求过大'},413),request);
        let body;try{body=JSON.parse(raw);}catch{return cors(json({error:'请求格式无效'},400),request);}
        const code=body.requestId?body.code:roomCode();
        if(!/^[A-Z2-9]{8}$/.test(code||''))return cors(json({error:'房间号无效'},400),request);
        response=await env.ROOMS.getByName(code).fetch(new Request(new URL('/init?code='+code,u),{method:'POST',headers:request.headers,body:raw}));
      }else{
        const match=u.pathname.match(/^\/rooms\/([A-Z2-9]{8})\/(join|socket)$/);
        if(!match)return cors(json({error:'地址不存在'},404),request);
        response=await env.ROOMS.getByName(match[1]).fetch(new Request(new URL('/'+match[2],u),request));
      }
    }catch{return cors(json({error:'服务暂时不可用，请重试'},503),request);}
    return response.status===101?response:cors(response,request);
  }
};

export class MahjongRoom extends DurableObject {
  constructor(ctx,env){
    super(ctx,env);this.ctx=ctx;this.room=null;
    this.ctx.blockConcurrencyWhile(async()=>{const saved=await ctx.storage.get('room');if(saved)this.room=new Room(saved);});
    this.ctx.setWebSocketAutoResponse(new WebSocketRequestResponsePair('ping','pong'));
  }
  async save(){
    if(!this.room)return;
    const authDeadlines=this.ctx.getWebSockets().map(ws=>ws.deserializeAttachment()).filter(a=>a?.seat==null).map(a=>a.authBy);
    // No await between the two writes: SQLite commits the state and wake-up
    // atomically, so a process restart cannot leave a saved game without a timer.
    await Promise.all([
      this.ctx.storage.put('room',this.room.serialize()),
      this.ctx.storage.setAlarm(Math.min(this.room.nextAlarm(),...authDeadlines))
    ]);
  }
  broadcast(){
    for(const ws of this.ctx.getWebSockets()){
      const a=ws.deserializeAttachment();if(a?.seat==null)continue;
      try{ws.send(JSON.stringify(this.room.view(a.seat)));}catch{}
    }
  }
  async fetch(request){
    try{
      const u=new URL(request.url);
      if(u.pathname==='/init'&&request.method==='POST'){
        const body=await request.text();if(body.length>2048)return json({error:'请求过大'},413);
        const input=JSON.parse(body);this.room??=new Room();const credential=this.room.init(u.searchParams.get('code'),input.name,input.requestId);await this.save();return json(credential);
      }
      if(!this.room||this.room.expired())return json({error:'房间不存在或已过期'},404);
      if(u.pathname==='/join'&&request.method==='POST'){
        const body=await request.text();if(body.length>2048)return json({error:'请求过大'},413);
        const input=JSON.parse(body),credential=this.room.join(input.name,input.requestId);await this.save();this.broadcast();return json(credential);
      }
      if(u.pathname==='/socket'&&request.headers.get('Upgrade')?.toLowerCase()==='websocket'){
        if(this.ctx.getWebSockets().length>=12)return json({error:'连接过多，请稍后重试'},429);
        const pair=new WebSocketPair(),[client,server]=Object.values(pair);this.ctx.acceptWebSocket(server);
        server.serializeAttachment({authBy:Date.now()+10000,seat:null,window:Date.now(),count:0});await this.save();
        return new Response(null,{status:101,webSocket:client});
      }return json({error:'地址不存在'},404);
    }catch(e){return json({error:e.message||'请求无效'},400);}
  }
  async webSocketMessage(ws,message){
    let a=ws.deserializeAttachment(),command;
    try{
      if(typeof message!=='string'||message.length>4096)throw Error('消息无效');
      if(Date.now()-a.window>10000){a.window=Date.now();a.count=0;}if(++a.count>60){ws.close(1008,'操作过于频繁');return;}
      command=JSON.parse(message);
      if(a.seat==null){
        if(command.type!=='auth')throw Error('请先验证座位');const seat=this.room.seat(command.token);
        if(seat<0){ws.close(1008,'座位凭证无效');return;}this.room.check();
        for(const old of this.ctx.getWebSockets())if(old!==ws&&old.deserializeAttachment()?.seat===seat){old.serializeAttachment({seat:null,authBy:0,window:Date.now(),count:0});old.close(4001,'该座位在另一页面打开');}
        a.seat=seat;ws.serializeAttachment(a);this.room.presence(seat,true);await this.save();this.broadcast();return;
      }
      ws.serializeAttachment(a);
      if(command.type==='command'){
        const ack=this.room.command(a.seat,command);await this.save();ws.send(JSON.stringify({type:'ack',...ack}));this.broadcast();
      }else if(command.type==='sync')ws.send(JSON.stringify(this.room.view(a.seat)));
      else throw Error('消息类型无效');
    }catch(e){
      ws.send(JSON.stringify({type:'error',id:command?.id,message:e.message||'操作失败'}));
      if(a?.seat!=null)ws.send(JSON.stringify(this.room.view(a.seat)));
    }
  }
  async webSocketClose(ws,code,reason,wasClean){
    const a=ws.deserializeAttachment();try{ws.close(code===1005?1000:code,'连接关闭');}catch{}
    if(a?.seat!=null&&this.room&&!this.ctx.getWebSockets().some(other=>other!==ws&&other.deserializeAttachment()?.seat===a.seat)){
      this.room.presence(a.seat,false);await this.save();this.broadcast();
    }
  }
  async webSocketError(ws){await this.webSocketClose(ws,1011,'连接错误',false);}
  async alarm(){
    if(!this.room)return;
    if(this.room.expired()){
      for(const ws of this.ctx.getWebSockets())ws.close(4004,'房间已过期');await this.ctx.storage.deleteAll();this.room=null;return;
    }
    for(const ws of this.ctx.getWebSockets()){
      const a=ws.deserializeAttachment();if(a?.seat==null&&Date.now()>=a.authBy)ws.close(1008,'认证超时');
    }
    const changed=this.room.tick();await this.save();if(changed)this.broadcast();
  }
}
