import {tileName,claimType,claimJokers,meldTiles,JOKER} from './engine.js?v=0.3.0';
export const variantHint = '幺鸡可代任意牌，可换出、碰、杠，不能打出，也不受定缺限制；其他缺门牌须先清完。整副胡牌（含碰杠）没有幺鸡，加 1 番。';
export function variantPicker(rules,disabled=false){
  return `<div class="variant-picker"><label for="variant-joker">麻将玩法<select id="variant-joker" ${disabled?'disabled':''}><option value="false" ${!rules.joker?'selected':''}>普通血战</option><option value="true" ${rules.joker?'selected':''}>幺鸡万能牌</option></select></label><label for="variant-swap">开局方式<select id="variant-swap" ${disabled?'disabled':''}><option value="true" ${rules.swap?'selected':''}>换三张后定缺</option><option value="false" ${!rules.swap?'selected':''}>直接定缺（不换三张）</option></select></label></div><p class="variant-hint">${rules.joker?variantHint:'血战到底，自摸加番，五番封顶。两种开局方式都保留定缺。'}</p>`;
}
export function claimLabel(action){const j=claimJokers(action);return ({hu:'胡牌',pong:'碰',kong:'杠'}[claimType(action)]||action)+(j?`（用${j}鸡）`:'');}
export function selfLabel(a){return a.type==='hu'?'自摸胡':a.type==='reclaim'?`换回幺鸡（${tileName(a.tile)}）`:`${a.type==='ankong'?'暗杠':'补杠'}${tileName(a.tile)}${a.jokers?`（用${a.jokers}鸡）`:''}`;}
export function meldCaption(m){const j=meldTiles(m).filter(t=>t===JOKER).length;return `${m.type==='pong'?'碰':'杠'}${tileName(m.tile)}${j?` · ${j}鸡` : ''}`;}
