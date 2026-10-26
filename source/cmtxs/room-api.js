// Admission is retryable only because every attempt carries the same secret ID.
export function admissionRequest(create,name,code){
  const bytes=crypto.getRandomValues(new Uint8Array(32));
  const requestId=Array.from(bytes,b=>b.toString(16).padStart(2,'0')).join('');
  const alphabet='ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  return {create,name,code:create?Array.from(crypto.getRandomValues(new Uint8Array(8)),b=>alphabet[b%32]).join(''):code,requestId};
}
export class RoomNetworkError extends Error {
  constructor(message='当前网络无法连接联机服务，请切换网络后重试。此次请求已保留，不会重复占座。'){super(message);this.name='RoomNetworkError';}
}
export function admissionError(error){
  if(error instanceof RoomNetworkError)return error.message;
  if(['AbortError','TimeoutError','TypeError'].includes(error?.name))return new RoomNetworkError().message;
  return error?.message||'暂时无法入座，请重试。';
}
export async function admitRoom(origins,request,{fetcher=fetch,timeoutMs=12000,onAttempt=()=>{},attempts=2}={}){
  for(let attempt=0;attempt<attempts;attempt++){
    const origin=origins[attempt%origins.length],controller=new AbortController();
    const timer=setTimeout(()=>controller.abort(),timeoutMs);onAttempt(attempt+1,attempts);
    try{
      const response=await fetcher(`${origin}${request.create?'/rooms':'/rooms/'+request.code+'/join'}`,{
        method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:request.name,requestId:request.requestId,code:request.code}),signal:controller.signal,cache:'no-store'
      });
      let body;try{body=await response.json();}catch{throw new RoomNetworkError();}
      if(!response.ok){
        if(response.status>=500||response.status===429)throw new RoomNetworkError();
        throw Object.assign(new Error(body.error||'暂时无法入座，请重试。'),{final:true});
      }
      if(typeof body.token!=='string'||body.token.length!==64||!Number.isInteger(body.seat))throw new RoomNetworkError();
      return {...body,server:origin};
    }catch(error){if(error.final)throw error;if(attempt===attempts-1)throw new RoomNetworkError();}
    finally{clearTimeout(timer);}
  }
  throw new RoomNetworkError();
}
