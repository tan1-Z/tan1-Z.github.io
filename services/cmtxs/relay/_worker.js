// Free alternative entry; the same authoritative room service owns all state.
const upstream='https://tingchuan-rooms.tingchuan-mahjong.workers.dev';
export default {
  async fetch(request){
    const url=new URL(request.url);
    if(url.pathname!=='/health'&&url.pathname!=='/rooms'&&!/^\/rooms\/[A-Z2-9]{8}\/(join|socket)$/.test(url.pathname))return new Response('听川联机服务',{status:200,headers:{'Content-Type':'text/plain; charset=utf-8'}});
    try{return await fetch(new Request(upstream+url.pathname+url.search,request));}
    catch{return Response.json({error:'联机服务暂时不可用，请重试。'},{status:503,headers:{'Access-Control-Allow-Origin':'https://tan1-z.github.io','Cache-Control':'no-store'}});}
  }
};
