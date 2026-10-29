import {tileName} from './engine.js?v=0.6.0';

// All game surfaces use the same 27 faces and cache revision.
export function tileFace(t){
  const name=(t%9+1)+['m','p','s'][Math.floor(t/9)];
  return `<img src="./assets/tiles/${name}.svg?v=0.3.4" alt="${tileName(t)}" draggable="false" width="40" height="56">`;
}
