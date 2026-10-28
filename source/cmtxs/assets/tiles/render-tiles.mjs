import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
const out=new URL('./',import.meta.url);
const glyphs=JSON.parse(await readFile(new URL('./tile-glyphs.json',import.meta.url),'utf8'));
const ink={green:'#075324',red:'#b51816',black:'#171a17'};
const path=(d,attrs='')=>`<path d="${d}" ${attrs}/>`;
const bamboo=(x,y,w,h,color='green')=>`<g transform="translate(${x} ${y}) scale(${w/12} ${h/28})" data-bamboo="${color}">${path('M6 1C1 1 1 5 3.1 7.1L3.1 12C.7 13.1 .7 15.1 3.1 16.2L3.1 21C.8 23.2 1.3 27 6 27S11.2 23.2 8.9 21L8.9 16.2C11.3 15.1 11.3 13.1 8.9 12L8.9 7.1C11 5 11 1 6 1Z',`fill="${ink[color]}" stroke="${ink[color]}" stroke-width="1" stroke-linejoin="round"`)}<path d="M5 4V11M5 17V23" fill="none" stroke="#f6f4e9" stroke-opacity=".85" stroke-width="1.1" stroke-linecap="round"/></g>`;
function bird(){return `<g data-bird="one-bamboo" fill="none" stroke="${ink.green}" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
<path d="M30 19Q26 16 29 13M32 19Q37 15 43 14M39 20L51 22 40 27"/>
<path d="M39 24C35 18 27 20 25 27C23 32 27 38 31 39L32 44C22 43 23 39 20 38C16 35 10 37 12 42C14 47 20 50 26 48C23 53 17 57 12 58L5 57 8 61 17 61C23 61 27 58 30 54C29 66 34 77 43 82C39 71 37 62 39 53C44 50 46 44 42 39C39 35 37 34 38 30C43 30 43 27 39 24Z" fill="#f7f5e9"/>
<path d="M27 48C31 49 36 48 38 43M32 43L33 37M33 53Q31 65 38 75M37 51Q36 65 44 76M26 50L22 57M18 54L15 48M38 45L48 46M46 44L49 42 51 44 49 47M31 28A3.5 3.5 0 1 0 31 27.9"/>
<path d="M21 39L17 35M22 33L25 33M23 27L21 25"/>
</g>`;}
function bambooFace(n){
 if(n===1)return bird();
 const b=(x,y,w,h,c='green')=>bamboo(x,y,w,h,c);
 if(n===2)return b(25,12,14,30)+b(25,47,14,30);
 if(n===3)return b(25,10,14,31,'red')+b(10,48,14,31)+b(40,48,14,31);
 if(n===4)return [11,39].flatMap(x=>[12,48].map(y=>b(x,y,14,30))).join('');
 if(n===5)return [8,43].flatMap(x=>[11,49].map(y=>b(x,y,12,29))).join('')+b(26,30,12,29,'red');
 if(n===6)return [8,26,44].flatMap(x=>[12,48].map(y=>b(x,y,12,30))).join('');
 if(n===7)return b(27,7,10,24,'red')+[9,27,45].flatMap(x=>[34,59].map(y=>b(x,y,10,24))).join('');
 if(n===8)return [15,52].map(y=>`<g transform="translate(0 ${y})" data-bamboo="green"><path d="M10 1L10 23Q10 26 14 23L29 7Q32 4 35 7L50 23Q54 26 54 22L54 1" fill="none" stroke="${ink.green}" stroke-width="6.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M10 2V22L29 5Q32 2 35 5L54 22V2" fill="none" stroke="#f7f5e9" stroke-width="1" stroke-linecap="round"/></g>`).join('');
 return [9,27,45].flatMap((x,c)=>[10,35,60].map(y=>b(x,y,10,24,c===1?'red':'green'))).join('');
}
function pip(x,y,r,color){return `<g data-pip="${color}" stroke="${ink[color]}" fill="none"><circle cx="${x}" cy="${y}" r="${r}" stroke-width="1.6"/><circle cx="${x}" cy="${y}" r="${r-3.2}" stroke-width="1.9"/><circle cx="${x}" cy="${y}" r="${Math.max(1,r-5.6)}" fill="${ink[color]}" stroke="none"/></g>`;}
function dotFace(n){
 if(n===1){let teeth=Array.from({length:20},(_,i)=>{let a=i*Math.PI/10;return `M${32+16*Math.cos(a)} ${47+16*Math.sin(a)}L${32+18.2*Math.cos(a+.045)} ${47+18.2*Math.sin(a+.045)}`;}).join('');return `<g data-pip="rosette" fill="none"><circle cx="32" cy="47" r="24" stroke="${ink.green}" stroke-width="2"/><circle cx="32" cy="47" r="20.3" stroke="${ink.green}" stroke-width="1.4"/><path d="${teeth}" stroke="${ink.green}" stroke-width="1.6"/><circle cx="32" cy="47" r="13.2" stroke="${ink.red}" stroke-width="2"/><circle cx="32" cy="47" r="9" stroke="${ink.red}" stroke-width="2"/><circle cx="32" cy="47" r="5.6" fill="${ink.red}"/></g>`;}
 const r=n===2?12:n<=5?10:n===6?9.7:n===7?9:n===8?8.8:8.3;
 const patterns={
 2:[[32,24,'green'],[32,66,'green']],
 3:[[15,18,'green'],[32,45,'red'],[49,72,'black']],
 4:[[16,23,'green'],[48,23,'black'],[16,67,'black'],[48,67,'green']],
 5:[[16,21,'green'],[48,21,'black'],[32,45,'red'],[16,69,'black'],[48,69,'green']],
 6:[[17,17,'green'],[47,17,'green'],[17,46,'red'],[47,46,'red'],[17,74,'red'],[47,74,'red']],
 7:[[15,13,'green'],[32,25,'green'],[49,37,'green'],[17,55,'red'],[47,55,'red'],[17,77,'red'],[47,77,'red']],
 8:[16,48].flatMap(x=>[13,34,55,76].map(y=>[x,y,'black'])),
 9:[14,32,50].flatMap(x=>[[x,17,'black'],[x,45,'red'],[x,73,'green']])
 };return patterns[n].map(([x,y,c])=>pip(x,y,r,c)).join('');
}
await mkdir(out,{recursive:true});
for(const suit of ['m','s','p'])for(let n=1;n<=9;n++){
 const label=n+({m:'万',s:'条',p:'筒'}[suit]);
 const face=suit==='m'?path(glyphs['一二三四伍六七八九'[n-1]],`fill="${ink.black}" stroke="${ink.black}" stroke-width=".55" stroke-linejoin="round"`)+path(glyphs['萬'],`fill="${ink.red}" stroke="${ink.red}" stroke-width=".45" stroke-linejoin="round"`):suit==='s'?bambooFace(n):dotFace(n);
 await writeFile(new URL(`${n}${suit}.svg`,out),`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 90" role="img" aria-label="${label}"><title>${label}</title>${face}</svg>\n`);
}
console.log(`Rendered 27 reference-style vector tile faces to ${fileURLToPath(out)}`);
