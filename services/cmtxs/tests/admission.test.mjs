import test from 'node:test';
import assert from 'node:assert/strict';
import {Room,secret} from '../room.js';
import {startServer} from '../local-server.mjs';
import {admissionRequest,admitRoom,admissionError} from '../../../source/cmtxs/room-api.js';

test('admission retries recover the same seat even after persistence and game start',()=>{
  const r=new Room(),hostId=secret(),joinId=secret(),host=r.init('ABCD2345','房主',hostId),friend=r.join('朋友',joinId);
  const snapshot=JSON.stringify(r.serialize());assert.deepEqual(r.init('ABCD2345','房主',hostId),host);assert.deepEqual(r.join('朋友',joinId),friend);assert.equal(JSON.stringify(r.serialize()),snapshot);
  r.command(0,{id:'ready-host',action:'ready'});r.command(1,{id:'ready-friend',action:'ready'});r.command(0,{id:'start',action:'start'});
  const restored=new Room(r.serialize());assert.deepEqual(restored.join('朋友',joinId),friend);assert.equal(restored.d.members.filter(Boolean).length,2);
  const wire=JSON.stringify(restored.view(0));assert.ok(!wire.includes(joinId));assert.ok(!wire.includes(hostId));
  assert.throws(()=>restored.init('ABCD2345','其他人',secret()),/已存在/);assert.throws(()=>restored.join('其他人',secret()),/已开始/);
});
test('legacy rooms stay joinable and invalid admission IDs consume no seat',()=>{
  const r=new Room();r.init('ABCD2345','房主');delete r.d.admissions;const old=new Room(r.serialize());
  assert.throws(()=>old.join('朋友','bad'),/无效/);assert.equal(old.d.members.filter(Boolean).length,1);assert.equal(old.join('朋友',secret()).seat,1);
});
test('a lost create or join HTTP response retries without creating duplicate rooms or seats',async()=>{
  const server=await startServer({port:0}),base='http://127.0.0.1:'+server.port;
  try{
    async function dropOnce(request){
      let calls=0,accepted;
      const result=await admitRoom([base],request,{timeoutMs:150,fetcher:async(url,options)=>{
        calls++;const response=await fetch(url,options);if(calls>1)return response;
        accepted=await response.clone().json();
        return new Promise((resolve,reject)=>{const abort=()=>reject(new DOMException('The user aborted a request.','AbortError'));if(options.signal.aborted)abort();else options.signal.addEventListener('abort',abort,{once:true});});
      }});
      assert.equal(calls,2);assert.equal(result.token,accepted.token);assert.equal(result.seat,accepted.seat);return result;
    }
    const host=await dropOnce(admissionRequest(true,'房主',''));
    const friend=await dropOnce(admissionRequest(false,'朋友',host.code));
    assert.equal(host.code,friend.code);assert.equal(server.rooms.size,1);assert.equal(server.rooms.get(host.code).d.members.filter(Boolean).length,2);
    const latest=await admitRoom([base],admissionRequest(false,'第三人',host.code));assert.equal(latest.seat,2);
  }finally{await server.close();}
});
test('blocked network retries a different entry and raw browser abort errors never reach users',async()=>{
  const request=admissionRequest(false,'朋友','ABCD2345'),attempted=[];
  await assert.rejects(admitRoom(['https://one.invalid','https://two.invalid'],request,{timeoutMs:5,fetcher:(url,{signal})=>{
    attempted.push(url);return new Promise((resolve,reject)=>signal.addEventListener('abort',()=>reject(new DOMException('The user aborted a request.','AbortError')),{once:true}));
  }}),error=>{assert.match(admissionError(error),/网络|联机/);assert.ok(!admissionError(error).includes('aborted'));return true;});
  assert.equal(attempted.length,2);assert.ok(attempted[1].startsWith('https://two.invalid'));
  assert.match(admissionError(new DOMException('The user aborted a request.','AbortError')),/网络/);
});
test('room full and started errors stop retries immediately and retain the server explanation',async()=>{
  let calls=0;await assert.rejects(admitRoom(['https://one.invalid'],admissionRequest(false,'朋友','ABCD2345'),{fetcher:async()=>{calls++;return Response.json({error:'房间已满'},{status:400});}}),/房间已满/);assert.equal(calls,1);
});
