import type { Metadata,Viewport } from 'next';
import type { ReactNode } from 'react';
import '../src/style.css';
import '../src/field.css';
export const metadata:Metadata={title:'Aviary · Home',description:'Live birds and recorded moments, privately at home.',manifest:'/manifest.webmanifest',icons:{icon:'/icon.svg',apple:'/icon-192.png'},appleWebApp:{capable:true,statusBarStyle:'black-translucent',title:'Aviary'}};
export const viewport:Viewport={width:'device-width',initialScale:1,viewportFit:'cover',themeColor:'#111313'};
export default function Layout({children}:{children:ReactNode}){return <html lang="en"><body>{children}</body></html>;}
