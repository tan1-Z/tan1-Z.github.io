// The joker stays a physical 1-bamboo. Substitution is used only for hand evaluation.
export const JOKER = 18;
const count = hand => {const c=Array(27).fill(0);for(const t of hand)c[t]++;return c;};
function meldsComplete(c,j,left,memo){
  const key=c.join(',')+'|'+j+'|'+left;
  if(memo.has(key))return memo.get(key);
  const i=c.findIndex(n=>n>0);
  if(i<0)return j===left*3;
  if(!left)return false;
  for(let n=Math.min(3,c[i]);n>=1;n--)if(3-n<=j){
    c[i]-=n;const ok=meldsComplete(c,j-3+n,left-1,memo);c[i]+=n;
    if(ok){memo.set(key,true);return true;}
  }
  const base=Math.floor(i/9)*9;
  for(let start=Math.max(base,i-2);start<=Math.min(base+6,i);start++){
    c[i]--;
    const other=[start,start+1,start+2].filter(t=>t!==i);
    function use(k,w){
      if(k===2)return meldsComplete(c,w,left-1,memo);
      const t=other[k];
      if(c[t]){c[t]--;const ok=use(k+1,w);c[t]++;if(ok)return true;}
      return w>0&&use(k+1,w-1);
    }
    const ok=use(0,j);c[i]++;
    if(ok){memo.set(key,true);return true;}
  }
  memo.set(key,false);return false;
}
export function canComplete(hand,open=0,wild=false){
  if(hand.length!==14-open*3)return false;
  const c=count(hand);if(c.some(n=>n>4))return false;
  const j=wild?c[JOKER]:0;if(wild)c[JOKER]=0;
  if(!open){const odd=c.reduce((n,v)=>n+v%2,0);if(odd<=j&&(j-odd)%2===0)return true;}
  const memo=new Map();
  for(let t=0;t<27;t++)for(let n=Math.min(2,c[t]);n>=1;n--)if(2-n<=j){
    c[t]-=n;const ok=meldsComplete(c,j-2+n,4-open,memo);c[t]+=n;if(ok)return true;
  }
  return j>=2&&meldsComplete(c,j-2,4-open,memo);
}
// Enumerating at most four jokers is bounded (27 choose-with-repetition 4).
// This path is used for scoring a completed hand, never for every AI candidate.
export function bestAssignment(hand,open,que,evaluate){
  const c=count(hand),j=c[JOKER];c[JOKER]=0;
  const natural=hand.filter(t=>t!==JOKER),assigned=[],memo=new Map();let best=null;
  const allowed=Array.from({length:27},(_,t)=>t).filter(t=>Math.floor(t/9)!==que);
  function walk(left,start){
    if(!left){
      const seven=!open&&c.every(n=>n%2===0);
      if(!seven){let ok=false;for(let t=0;t<27;t++)if(c[t]>=2){c[t]-=2;ok=meldsComplete(c,0,4-open,memo);c[t]+=2;if(ok)break;}if(!ok)return;}
      const result=evaluate([...natural,...assigned]);
      if(!best||result.rawFan>best.rawFan)best={...result,substitutions:[...assigned]};return;
    }
    for(let k=start;k<allowed.length;k++){const t=allowed[k];c[t]++;assigned.push(t);walk(left-1,k);assigned.pop();c[t]--;}
  }
  walk(j,0);return best;
}
