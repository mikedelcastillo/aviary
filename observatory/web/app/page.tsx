'use client';
import dynamic from 'next/dynamic';
const Aviary=dynamic(()=>import('../src/App'),{ssr:false,loading:()=> <div className="boot"><img src="/icon.svg" alt="Aviary"/><p>Opening Aviary…</p></div>});
export default function Page(){return <Aviary/>;}
