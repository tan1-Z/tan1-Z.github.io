export const EMOTE_COOLDOWN_MS=3000;
export const EMOTE_DURATION_MS=4200;
export const EMOTES=Object.freeze([
  ['steady','稳得很','牌友'],['lucky','牌运拉满','牌友'],['hopeful','这把有戏','牌友'],['draw-good','摸张好牌','牌友'],['thinking','让我想想','牌友'],['shy-win','麻胡，献丑了','牌友'],
  ['chicken','幺鸡护体','原创'],['kong-flower','杠上开花','原创'],['no-deal-in','拒绝点炮','原创']
].map(([id,label,group])=>Object.freeze({id,label,group,src:`./assets/emotes/${id}.webp?v=0.6.0`})));
export const getEmote=id=>EMOTES.find(e=>e.id===id);
export const isEmoteId=id=>typeof id==='string'&&EMOTES.some(e=>e.id===id);

// A display-only inbox. It never enters a game, replay, or action queue.
export class EmoteFeed {
  constructor({now=Date.now}={}){this.now=now;this.active=new Map();this.seen=new Set();this.muted=false;}
  receive(event,offset=0){
    if(!event||!Number.isInteger(event.seat)||event.seat<0||event.seat>3||!isEmoteId(event.sticker)||typeof event.id!=='string'||!/^[\w-]{1,80}$/.test(event.id)||!Number.isFinite(event.at)||!Number.isFinite(event.expiresAt)||!Number.isFinite(offset))return false;
    const key=event.seat+':'+event.id,at=event.at-offset,until=event.expiresAt-offset,now=this.now();
    if(this.seen.has(key)||until<=now||at>now+5000||until<=at||until-at>EMOTE_DURATION_MS)return false;
    this.seen.add(key);if(this.seen.size>64)this.seen.delete(this.seen.values().next().value);
    if(this.muted)return false;
    this.active.set(event.seat,{...event,key,at,until:Math.min(until,now+EMOTE_DURATION_MS)});return true;
  }
  current(){for(const [seat,e] of this.active)if(e.until<=this.now())this.active.delete(seat);return [...this.active.values()];}
  setMuted(value){this.muted=!!value;if(this.muted)this.active.clear();}
  clear(){this.active.clear();this.seen.clear();}
}
