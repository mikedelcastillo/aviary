import {VideoRTC} from './vendor/video-rtc.js';
class AviaryVideo extends VideoRTC {
  constructor(){super();this.mode='mse';this.media='video';this.pcConfig={iceServers:[]};this.RECONNECT_TIMEOUT=4000;this.DISCONNECT_TIMEOUT=1000;}
  oninit(){super.oninit();this.video.muted=true;this.video.autoplay=true;this.video.setAttribute('playsinline','');this.video.disablePictureInPicture=false;for(const name of ['playing','waiting','error'])this.video.addEventListener(name,()=>this.dispatchEvent(new CustomEvent('playback',{detail:name})));}
}
customElements.define('aviary-video',AviaryVideo);
