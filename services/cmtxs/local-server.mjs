import http from 'node:http';
import {WebSocketServer} from 'ws';
import {readFile,mkdir,writeFile,readdir,rename} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Room,roomCode} from './room.js';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../../source/cmtxs');
export async function startServer({port=4173,dataDir=null,timeout=30000,botDelay=700}={}){
  const rooms=new Map(),sockets=new Map();
  if(dataDir){await mkdir(dataDir,{recursive:true});for(const file of await readdir(dataDir))if(/^[A-Z2-9]{8}\.json$/.test(file)){const r=new Room(JSON.parse(await readFile(path.join(dataDir,file),'utf8')),{timeout,botDelay});if(!r.expired()){r.d.members.forEach(m=>{if(m)m.online=false;});rooms.set(r.d.code,r);}}}
  const persist=async r=>{if(dataDir){const file=path.join(dataDir,r.d.code+'.json');await writeFile(file+'.tmp',JSON.stringify(r.serialize()));await rename(file+'.tmp',file);}};
  const broadcast=r=>{for(const [ws,a] of sockets)if(a.room===r&&a.seat!=null&&ws.readyState===1)ws.send(JSON.stringify(r.view(a.seat)));};
  let queue=Promise.resolve();const serial=fn=>{queue=queue.then(fn).catch(e=>console.error(e.message));};
  const server=http.createServer((req,res)=>serial(async()=>{
    const send=(status,body)=>{res.writeHead(status,{'Content-Type':'application/json','Cache-Control':'no-store'});res.end(JSON.stringify(body));};
    try{
      const u=new URL(req.url,'http://localhost');
      if(req.method==='GET'&&u.pathname==='/health')return send(200,{ok:true,protocol:1,version:'0.3.0'});
      if(req.method==='POST'){
        let raw='';for await(const chunk of req){raw+=chunk;if(raw.length>2048)return send(413,{error:'请求过大'});}
        const body=JSON.parse(raw),m=u.pathname.match(/^\/rooms\/([A-Z2-9]{8})\/join$/);let r,credential;
        if(u.pathname==='/rooms'){
          let code=body.requestId?body.code:null;if(code&&!/^[A-Z2-9]{8}$/.test(code))throw Error('房间号无效');
          if(!code)do{code=roomCode();}while(rooms.has(code));
          r=rooms.get(code)||new Room(null,{timeout,botDelay});credential=r.init(code,body.name,body.requestId);rooms.set(code,r);
        }
        else if(m){r=rooms.get(m[1]);if(!r)return send(404,{error:'房间不存在'});credential=r.join(body.name,body.requestId);}
        else return send(404,{error:'地址不存在'});
        await persist(r);broadcast(r);return send(200,credential);
      }
      const relative=decodeURIComponent(u.pathname==='/'?'/index.html':u.pathname);const file=path.resolve(root,'.'+relative);
      if(!file.startsWith(root+path.sep))return send(403,{error:'禁止访问'});
      const content=await readFile(file),type={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.svg':'image/svg+xml','.json':'application/json'}[path.extname(file)]||'application/octet-stream';
      res.writeHead(200,{'Content-Type':type,'Cache-Control':'no-store'});res.end(content);
    }catch(e){send(e.code==='ENOENT'?404:400,{error:e.message});}
  }));
  const wss=new WebSocketServer({noServer:true,maxPayload:4096});
  server.on('upgrade',(req,socket,head)=>{
    const match=req.url.match(/^\/rooms\/([A-Z2-9]{8})\/socket$/),r=match&&rooms.get(match[1]);
    const origin=req.headers.origin;const valid=!origin||new URL(origin).hostname==='localhost'||new URL(origin).hostname==='127.0.0.1';
    if(!r||r.expired()||!valid){socket.write('HTTP/1.1 403 Forbidden\r\n\r\n');socket.destroy();return;}
    if([...sockets.values()].filter(a=>a.room===r).length>=12){socket.destroy();return;}
    wss.handleUpgrade(req,socket,head,ws=>wss.emit('connection',ws,r));
  });
  wss.on('connection',(ws,r)=>{
    const a={room:r,seat:null,count:0,window:Date.now()};sockets.set(ws,a);const authTimer=setTimeout(()=>{if(a.seat==null)ws.close(1008,'认证超时');},10000);
    ws.on('message',buffer=>serial(async()=>{
      let c;try{
        const text=buffer.toString();if(text==='ping'){ws.send('pong');return;}
        if(Date.now()-a.window>10000){a.window=Date.now();a.count=0;}if(++a.count>60){ws.close(1008,'操作过于频繁');return;}
        c=JSON.parse(text);
        if(a.seat==null){
          if(c.type!=='auth')throw Error('请先验证座位');const seat=r.seat(c.token);if(seat<0){ws.close(1008,'座位凭证无效');return;}r.check();
          for(const [other,b] of sockets)if(other!==ws&&b.room===r&&b.seat===seat){b.seat=null;other.close(4001,'该座位在另一页面打开');}
          a.seat=seat;clearTimeout(authTimer);r.presence(seat,true);await persist(r);broadcast(r);return;
        }
        if(c.type==='command'){const ack=r.command(a.seat,c);await persist(r);ws.send(JSON.stringify({type:'ack',...ack}));broadcast(r);}
        else if(c.type==='sync')ws.send(JSON.stringify(r.view(a.seat)));
        else throw Error('消息类型无效');
      }catch(e){ws.send(JSON.stringify({type:'error',id:c?.id,message:e.message}));if(a.seat!=null)ws.send(JSON.stringify(r.view(a.seat)));}
    }));
    ws.on('close',()=>serial(async()=>{clearTimeout(authTimer);sockets.delete(ws);if(a.seat!=null){r.presence(a.seat,false);await persist(r);broadcast(r);}}));
    ws.on('error',()=>{});
  });
  const timer=setInterval(()=>serial(async()=>{for(const r of rooms.values())if(r.tick()){await persist(r);broadcast(r);}}),100);timer.unref();
  await new Promise(resolve=>server.listen(port,'127.0.0.1',resolve));
  return {port:server.address().port,rooms,close:async()=>{clearInterval(timer);const stopped=new Promise(resolve=>server.close(resolve));for(const ws of sockets.keys())ws.terminate();await new Promise(resolve=>wss.close(resolve));await queue;await stopped;}};
}
if(process.argv[1]===fileURLToPath(import.meta.url)){
  const s=await startServer({port:Number(process.env.PORT||4173),dataDir:process.env.ROOM_DATA_DIR||path.resolve('.room-data')});console.log(`听川联机服务 http://127.0.0.1:${s.port}/rooms.html`);
}
