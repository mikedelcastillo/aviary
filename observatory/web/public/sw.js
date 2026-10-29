// Never cache API responses, photos, live frames or authentication responses.
const CACHE='aviary-shell-v2';
self.addEventListener('install',event=>{event.waitUntil(caches.open(CACHE).then(c=>c.addAll(['/icon.svg','/manifest.webmanifest'])));self.skipWaiting()});
self.addEventListener('activate',event=>event.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(k=>k.startsWith('aviary-shell-')&&k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim())));
self.addEventListener('fetch',event=>{const u=new URL(event.request.url);if(u.origin===self.location.origin&&['/icon.svg','/manifest.webmanifest'].includes(u.pathname))event.respondWith(caches.match(event.request).then(r=>r||fetch(event.request)))});
