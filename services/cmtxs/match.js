const copy=x=>structuredClone(x);
export const matchRounds=n=>[0,4,8].includes(n);
export function newMatch(rounds=0){if(!matchRounds(rounds))throw Error('赛制应为自由对局、4局或8局');return {id:crypto.randomUUID(),rounds,started:false,results:[]};}
export const matchComplete=m=>m.results.length>=(m.rounds||32);
export function recordRound(m,g,roundId){
  if(g?.phase!=='ended'||m.results.some(r=>r.id===roundId))return false;
  if(matchComplete(m))return false;
  m.started=true;
  m.results.push({id:roundId,number:m.results.length+1,rules:copy(g.rules),players:g.players.map((p,seat)=>({seat,name:p.name,score:p.score,won:p.won,fan:p.win?.fan??0,self:!!p.win?.self,from:p.win?.from??null,tags:copy(p.win?.tags||[]),winPoints:g.ledger.filter(x=>x.to===seat&&x.kind==='win').reduce((n,x)=>n+x.amount,0),fires:g.players.filter(q=>q.won&&!q.win?.self&&q.win?.from===seat).length}))});
  return true;
}
export function matchView(m){
  const totals=Array(4).fill(0),wins=Array(4).fill(0),fires=Array(4).fill(0),selfs=Array(4).fill(0);let biggest=null,comeback=null;
  for(const r of m.results){
    const before=[...totals],lead=Math.max(...before);
    for(const p of r.players){totals[p.seat]+=p.score;wins[p.seat]+=p.won?1:0;fires[p.seat]+=p.fires;selfs[p.seat]+=p.won&&p.self?1:0;
      if(p.won&&(!biggest||p.fan>biggest.fan||p.fan===biggest.fan&&p.winPoints>biggest.points))biggest={seat:p.seat,round:r.number,fan:p.fan,points:p.winPoints};}
    const afterLead=Math.max(...totals);
    for(let seat=0;seat<4;seat++){const gap=lead-before[seat];if(gap>0&&totals[seat]===afterLead&&(!comeback||gap>comeback.gap))comeback={seat,round:r.number,gap};}
  }
  const ranks=totals.map(n=>1+totals.filter(x=>x>n).length),maxFires=Math.max(...fires);
  return {id:m.id,rounds:m.rounds,started:m.started,completed:matchComplete(m),played:m.results.length,totals,ranks,wins,fires,selfs,biggest,comeback,mostFires:maxFires?fires.flatMap((n,s)=>n===maxFires?[s]:[]):[],results:copy(m.results)};
}
