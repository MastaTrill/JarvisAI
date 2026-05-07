// Basic service worker for PWA
const CACHE_NAME = 'jarvis-ai-v1';
const urlsToCache = [
  '/',
  '/static/dashboard/index.html',
  '/static/manifest.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then((response) => response || fetch(event.request))
  );
});