export interface Observation {
  id: string; time: string; camera: string; bird: string; confidence: number | null;
  note: string; source: string; activity: string; bbox: number[]; image: string | null; favorite?: boolean;
}
export interface BirdProfile {
  id: string; name: string; species: string; portrait?: string | null;
  last: Observation | null; today: number; visits: Record<string,number>;
  color?: string; description?: string; sex?:string;
}
export interface Camera {
  id: string; name: string; online: boolean; age: number|null; privacy: string;
  ptz_supported?: boolean|null;
  video_ready?: boolean; video_reasons?: string[];
  capture_age?: number|null; recorder_age?: number|null;
  detections?: {bird:string;confidence:number;bbox:number[];track:number;identified:boolean}[];
  zone_views?: Record<string,{valid:boolean;reason:string}>;
  inference_ms?: number; reconnects?: number; playlist_age?: number|null; segments?: number; recording_mib?: number;
}
export interface Overview {birds:BirdProfile[];cameras:Camera[];archive:{count:number;first:string|null;last:string|null};timezone:string;}
export interface Zone {id:string;camera:string;name:string;kind:string;rect:[number,number,number,number];anchor?:string;}
export interface ChatMessage {role:string;text:string;time?:string;evidence?:Observation[];}
export interface ClipAnalysis {human_reviewed?:boolean;visible_evidence?:string[];uncertainty?:string;model?:string;latency_seconds?:number;}
export interface Clip {id:string;time:string;camera:string;duration:number;video:string;poster:string;status:string;caption:string;favorite:boolean;analysis:ClipAnalysis;birds:string[];}
export interface Stats {
  services:{name:string;state:string;memory_mib:number|null;cpu_cores:number|null;restarts:number}[];
  cameras:Camera[];coverage:{hour:string;count:number}[];events:{events:number};disk_free_gib:number;
  decisions:{status:string;count:number}[];archive_engine:string;db_mib:number;curator_model:string;
  auto_publish:boolean;host_load:number[];cpu_threads:number;last_judge:{caption:string;status:string;analysis:ClipAnalysis}|null;
}
export const errorMessage=(error:unknown)=>error instanceof Error?error.message:'Connection interrupted. Please try again.';
declare global {interface HTMLVideoElement {webkitEnterFullscreen?:()=>void;}}
