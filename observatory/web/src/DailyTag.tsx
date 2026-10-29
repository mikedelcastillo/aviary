'use client';
import {useEffect, useRef, useState} from 'react';
import type {PointerEvent} from 'react';
import {ArrowLeft, ArrowRight, Check, HelpCircle, Pencil, Plus, Trash2, Undo2, Video, ZoomIn} from 'lucide-react';
import type {Camera} from './types';
import {errorMessage} from './types';
import './learning.css';

type Box = {id:string; cx:number; cy:number; w:number; h:number; label:string|null; confidence?:number};
type Lighting = 'day'|'ir';
type Card = {id:string; camera:string; time:string; image:string; video:string|null; frame_seconds:number; lighting:Lighting; reason:string; suggestions:Box[]; category:string; imported:boolean};
type Queue = {cards:Card[]; today:number; goal:number; confirmed_frames:number; pending:number; labels:Record<Lighting|'phone', {name:string; kind:string}[]>};
type Review = {boxes:Box[]; lighting:Lighting; complete:boolean};
type Saved = {answer:string; index:number; outcome:string; review:Review};

async function request<T>(path:string, body?:unknown):Promise<T> {
 const response = await fetch('/api/learn'+path, body === undefined ? {cache:'no-store'} : {
  method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body),
 });
 const data = await response.json();
 if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : 'Could not save your answer. Please try again.');
 return data;
}

export default function DailyTag({cameras, back}:{cameras:Camera[]; back:()=>void}) {
 const [data, setData] = useState<Queue|null>(null), [index, setIndex] = useState(0);
 const [boxes, setBoxes] = useState<Box[]>([]), [selected, setSelected] = useState<string|null>(null);
 const [lighting, setLighting] = useState<Lighting>('day'), [complete, setComplete] = useState(false);
 const [drawing, setDrawing] = useState<string|null>(null), [draft, setDraft] = useState<Box|null>(null);
 const [zoom, setZoom] = useState(1), [context, setContext] = useState(false), [imageReady, setImageReady] = useState(false);
 const [busy, setBusy] = useState(false), [loading, setLoading] = useState(true), [error, setError] = useState('');
 const [extra, setExtra] = useState(false), [last, setLast] = useState<Saved|null>(null);
 const origin = useRef<{x:number; y:number}|null>(null), video = useRef<HTMLVideoElement>(null), still = useRef<HTMLImageElement>(null);
 const frameScroll = useRef<HTMLDivElement>(null), heading = useRef<HTMLDivElement>(null);
 const restore = useRef<Review|null>(null), saving = useRef(false);
 const card = data?.cards[index];
 const goalReached = !!data && data.today >= data.goal;
 const showCard = card && (extra || !goalReached);

 async function load(more=false) {
  setError(''); setLoading(true);
  try {setData(await request<Queue>(more ? '?more=true' : '')); setIndex(0); setLast(null); setExtra(more)}
  catch (e) {setError(errorMessage(e))}
  finally {setLoading(false)}
 }
 useEffect(() => {void load()}, []);
 useEffect(() => {
  if (!card) return;
  const review = restore.current;
  restore.current = null;
  const initialBoxes = review?.boxes || card.suggestions.map(b => ({...b, label:null}));
  setBoxes(initialBoxes); setSelected(initialBoxes[0]?.id || null);
  setLighting(review?.lighting || card.lighting); setComplete(review?.complete || false);
  setContext(false); setZoom(1); setDraft(null); setDrawing(null);
  setImageReady(!!still.current?.complete && still.current.naturalWidth > 0); origin.current = null;
  frameScroll.current?.scrollTo(0, 0);
  if (index > 0) heading.current?.scrollIntoView({block:'start', behavior:'smooth'});
 }, [card?.id]);
 useEffect(() => {
  const scroll = frameScroll.current;
  if (!scroll) return;
  const focus = boxes.find(b => b.id === selected);
  scroll.scrollTo({left:(focus?.cx ?? .5)*scroll.scrollWidth-scroll.clientWidth/2,
                   top:(focus?.cy ?? .5)*scroll.scrollHeight-scroll.clientHeight/2});
 }, [zoom]);

 function cancelDrawing() {origin.current = null; setDraft(null); setDrawing(null)}
 function startDrawing(id:string) {
  if (drawing === id) {cancelDrawing(); return}
  origin.current = null; setDraft(null); setDrawing(id); setComplete(false);
  // Redraw lives below the picture on phones. Keep the canvas clear of the
  // fixed header/navigation when bringing it back into view.
  frameScroll.current?.scrollIntoView({block:'center', behavior:'instant'});
 }
 function point(event:PointerEvent<SVGSVGElement>) {
  const r = event.currentTarget.getBoundingClientRect();
  return {x:Math.max(0, Math.min(1, (event.clientX-r.left)/r.width)), y:Math.max(0, Math.min(1, (event.clientY-r.top)/r.height))};
 }
 function rectangle(event:PointerEvent<SVGSVGElement>):Box|null {
  const o = origin.current;
  if (!o) return null;
  const p = point(event);
  return {id:'draft', cx:(p.x+o.x)/2, cy:(p.y+o.y)/2, w:Math.abs(p.x-o.x), h:Math.abs(p.y-o.y), label:null};
 }
 function down(event:PointerEvent<SVGSVGElement>) {
  if (!drawing || busy || !event.isPrimary) return;
  // A tap sets one corner; a second tap finishes it. Dragging also works.
  if (!origin.current) origin.current = point(event);
  event.currentTarget.setPointerCapture(event.pointerId);
  setComplete(false);
 }
 function move(event:PointerEvent<SVGSVGElement>) {
  if (drawing && origin.current && event.isPrimary) setDraft(rectangle(event));
 }
 function up(event:PointerEvent<SVGSVGElement>) {
  if (!drawing || busy || !event.isPrimary) return;
  const rect = rectangle(event);
  if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  if (!rect || rect.w <= .003 || rect.h <= .003) return;
  const existing = boxes.find(b => b.id === drawing);
  const box = {...rect, id:existing?.id || crypto.randomUUID(), label:existing?.label || null};
  setBoxes(old => existing ? old.map(b => b.id === existing.id ? box : b) : [...old, box]);
  setSelected(box.id); cancelDrawing();
 }
 async function save(outcome:'annotated'|'empty'|'uncertain'|'skip') {
  if (!card || saving.current) return;
  saving.current = true; setBusy(true); setError('');
  const review = {boxes:boxes.map(b => ({...b})), lighting, complete};
  try {
   const result = await request<{answer_id:string}>('/'+card.id+'/answer', {
    outcome, lighting, complete, boxes:outcome === 'empty' ? [] : boxes,
   });
   setLast({answer:result.answer_id, index, outcome, review}); setIndex(i => i+1);
   setData(d => d ? {...d, pending:Math.max(0, d.pending-1), today:d.today+(outcome === 'skip' ? 0 : 1),
    confirmed_frames:d.confirmed_frames+(['annotated','empty'].includes(outcome) ? 1 : 0)} : d);
  } catch (e) {setError(errorMessage(e))}
  finally {saving.current = false; setBusy(false)}
 }
 async function undo() {
  if (!last || saving.current) return;
  saving.current = true; setBusy(true); setError('');
  try {
   await request('/answers/'+last.answer+'/undo', {});
   restore.current = last.review;
   setData(d => d ? {...d, pending:d.pending+1, today:Math.max(0, d.today-(last.outcome === 'skip' ? 0 : 1)),
    confirmed_frames:Math.max(0, d.confirmed_frames-(['annotated','empty'].includes(last.outcome) ? 1 : 0))} : d);
   setIndex(last.index); setLast(null);
  } catch (e) {setError(errorMessage(e))}
  finally {saving.current = false; setBusy(false)}
 }
 const ready = imageReady && complete && !drawing && boxes.every(b => !!b.label);
 const labels = data?.labels[lighting === 'ir' ? 'ir' : card?.category === 'phone' ? 'phone' : 'day'] || [];
 const selectedBox = boxes.find(b => b.id === selected);

 return <section className="daily-tag">
  <div className="learning-intro" ref={heading}><div><p className="eyebrow">A LITTLE AT A TIME</p><h2>Daily five</h2></div><span className="learning-count">{Math.min(data?.today || 0, data?.goal || 5)} / {data?.goal || 5}</span></div>
  <div className="learning-progress" role="progressbar" aria-label="Reviewed today" aria-valuenow={Math.min(data?.today || 0, 5)} aria-valuemin={0} aria-valuemax={5}><i style={{width:Math.min(100, (data?.today || 0)*20)+'%'}}/></div>
  <p className="learning-note">A few useful corrections. Repeats are filtered out; some frames are chosen without relying on a bird detection.</p>
  {error && <p className="learning-error" role="alert">{error}</p>}
  {loading ? <p role="status">Finding a few frames…</p> : showCard ? <article className="learning-card" key={card.id}>
   <fieldset className="learning-review" disabled={busy}>
    <legend className="sr-only">Review the full frame</legend>
    <div className="learning-meta"><span>{card.imported ? 'From your image collection' : cameras.find(c => c.id === card.camera)?.name || 'Camera'} · {new Date(card.time).toLocaleString(undefined, {month:'short', day:'numeric', hour:'numeric', minute:'2-digit'})}</span><span>{card.reason === 'audit' ? 'Detection check' : card.reason === 'uncertain' ? 'Uncertain boxes' : card.imported ? 'Fresh review' : 'A different view'}</span></div>
    <div className="learning-tools">
     <button disabled={!imageReady || boxes.length >= 40} className={drawing === 'new' ? 'active' : ''} onClick={() => startDrawing('new')}><Plus size={17}/>{drawing === 'new' ? 'Cancel drawing' : 'Add a missed bird'}</button>
     <button disabled={!imageReady} aria-label={`Zoom ${zoom} times. Tap to change zoom.`} onClick={() => {cancelDrawing(); setZoom(z => z === 1 ? 2 : z === 2 ? 3 : 1)}}><ZoomIn size={17}/>{zoom}×</button>
     <button aria-label="Undo last submitted frame" disabled={!last} onClick={() => void undo()}><Undo2 size={17}/></button>
    </div>
    <div className="learning-scroll" ref={frameScroll}><div className="learning-frame" style={{width:zoom*100+'%'}}>
     <img ref={still} src={card.image} alt="Full frame to review for birds" draggable={false} onLoad={() => setImageReady(true)} onError={() => {setImageReady(false); setError('This frame could not load. Reload Daily five before confirming birds or an empty frame.')}}/>
     <svg viewBox="0 0 1 1" preserveAspectRatio="none" style={{touchAction:drawing ? 'none' : 'pan-x pan-y'}} onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={() => {origin.current = null; setDraft(null)}} aria-label="Bird annotation canvas">
      {[...boxes, ...(draft ? [draft] : [])].map((box, i) => <g key={box.id} onPointerDown={event => {if (!drawing && !busy) {event.stopPropagation(); setSelected(box.id)}}}>
       <rect x={box.cx-box.w/2} y={box.cy-box.h/2} width={box.w} height={box.h} fill={selected === box.id ? '#bad3bd28' : 'transparent'} stroke={box.label ? '#b4ddb6' : '#e6c58b'} strokeWidth="2" vectorEffect="non-scaling-stroke" strokeDasharray={box.label ? undefined : '6 3'}/>
       <text x={box.cx-box.w/2+.005} y={Math.max(.026, box.cy-box.h/2+.026)} fill="#fff" fontSize=".027" stroke="#000" strokeWidth=".002" paintOrder="stroke">{box.id === 'draft' ? '+' : i+1}</text>
      </g>)}
     </svg>
    </div></div>
    {drawing ? <p className="learning-draw-hint" role="status">Tap two opposite corners around the whole bird, or drag a box. <button onClick={cancelDrawing}>Cancel</button></p> : <p className="learning-note learning-zoom-hint">{zoom > 1 ? 'Drag to look around the enlarged frame. Zoom back to 1× to check the whole scene.' : 'Zoom in to inspect small birds. Add or redraw a box wherever needed.'}</p>}
    <div className="learning-context">
     {card.video ? <button onClick={() => setContext(!context)}><Video size={17}/>{context ? 'Hide video' : 'Watch nearby video'}</button> : <span className="learning-note">Original still · no video available</span>}
     <label>Lighting <select value={lighting} onChange={e => {setLighting(e.target.value as Lighting); setBoxes(b => b.map(x => ({...x, label:null}))); setComplete(false)}}><option value="day">Day / color</option><option value="ir">Infrared</option></select></label>
    </div>
    {context && card.video && <video className="learning-video" ref={video} src={card.video} controls playsInline preload="metadata" onLoadedMetadata={() => {if (video.current) video.current.currentTime = Math.max(0, card.frame_seconds-3)}} onError={() => setError('Video context is unavailable. You can still review the still frame, or choose Can’t tell.')}/>}
    <p className="learning-note">Label the still frame above. {card.video && 'Video gives context. '}Dashed boxes are unconfirmed suggestions; look outside them too.</p>
    <div className="learning-boxes">{boxes.map((box, i) => <button className={selected === box.id ? 'active' : ''} key={box.id} onClick={() => {cancelDrawing(); setSelected(box.id)}}>{i+1} · {box.label?.replaceAll('_', ' ') || 'Who is this?'}</button>)}{!boxes.length && <p>No boxes yet. Add any birds the detector missed.</p>}</div>
    {selectedBox && <div className="learning-labels">
     <div className="learning-label-heading"><strong>Name this bird</strong><div>
      <button className={drawing === selected ? 'active' : ''} onClick={() => startDrawing(selectedBox.id)}><Pencil size={16}/>Redraw</button>
      <button aria-label="Remove selected box" onClick={() => {cancelDrawing(); setBoxes(b => b.filter(x => x.id !== selected)); setSelected(boxes.find(b => b.id !== selected)?.id || null); setComplete(false)}}><Trash2 size={16}/>Remove</button>
     </div></div>
     <div className="learning-label-options">{labels.map(label => <button className={selectedBox.label === label.name ? 'active' : ''} key={label.name} onClick={() => {cancelDrawing(); setBoxes(b => b.map(x => x.id === selected ? {...x, label:label.name} : x)); setComplete(false); const next = boxes.find(b => b.id !== selected && !b.label); if (next) setSelected(next.id)}}>{label.name.replaceAll('_', ' ')}</button>)}</div>
     <p className="learning-note">{lighting === 'ir' ? 'Species only in infrared. Don’t guess identity from color that isn’t visible.' : 'Know it is a bird, but not which one? Use unknown bird. If you cannot check all the birds, choose Can’t tell.'}</p>
    </div>}
    <label className="learning-complete"><input type="checkbox" checked={complete} disabled={!imageReady || !!drawing} onChange={e => setComplete(e.target.checked)}/><span>{boxes.length ? 'I checked the whole frame and included every visible bird.' : 'I checked the whole frame. There are no birds.'}</span></label>
    <div className="learning-actions"><button onClick={() => void save('uncertain')}><HelpCircle size={18}/>Can’t tell</button><button className="primary" disabled={!ready} onClick={() => void save(boxes.length ? 'annotated' : 'empty')}><Check size={18}/>{busy ? 'Saving…' : boxes.length ? 'Save & next' : 'Confirm empty'}</button></div>
    <p className="learning-note learning-action-note">Can’t tell counts for today and stays out of training.</p>
    <button className="learning-skip" onClick={() => void save('skip')}>Set this one aside <ArrowRight size={15}/></button>
   </fieldset>
  </article> : data ? <div className="learning-done" role="status">
   <Check size={36}/><h2>{goalReached ? 'Your daily five are done.' : data.today ? 'That helps. Thank you.' : 'Nothing useful waiting yet.'}</h2>
   <p>{data.today ? `${data.today} reviewed today. Uncertain answers stay out of training.` : 'The sampler is looking for distinct frames, including frames without detections.'}</p>
   <button className="primary" onClick={back}><ArrowLeft size={17}/>Back to live</button>
   {data.pending > 0 && <button className="secondary" disabled={busy} onClick={() => void load(goalReached)}>{goalReached ? 'Five more, if you feel like it' : 'Continue daily five'}</button>}
   {last && <button disabled={busy} onClick={() => void undo()}><Undo2 size={17}/>Undo last answer</button>}
  </div> : <button onClick={() => void load()}>Try again</button>}
  <div className="learning-foot"><span>{data?.confirmed_frames || 0} confirmed frames</span><a href="/api/learn/export">Export for the annotation project</a></div>
 </section>;
}
