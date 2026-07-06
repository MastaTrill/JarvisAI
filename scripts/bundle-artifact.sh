#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('📦 Building Jarvis Dashboard...');

// Install bundling dependencies if needed
try {
  execSync('npm list parcel', { stdio: 'ignore' });
} catch (e) {
  console.log('Installing parcel bundler...');
  execSync('npm install -D parcel@latest', { stdio: 'inherit' });
}

// Build with parcel
console.log('Building with Parcel...');
execSync('parcel build index.html --dist-dir ./dist --public-url ./', { 
  stdio: 'inherit' 
});

// Read the built HTML
let html = fs.readFileSync('dist/index.html', 'utf8');

// Inline CSS
const css = fs.readFileSync('dist/styles.css', 'utf8');
html = html.replace(
  /<link rel="stylesheet" href="([^"]+)" \/>/,
  `<style>$1</style>`
);

// Inline JS
const js = fs.readFileSync('dist/index.js', 'utf8');
html = html.replace(
  /<script type="module" src="([^"]+)"><\/script>/,
  `<script>$1</script>`
);

// Clean up asset references
html = html.replace(/<link[^>]*>/g, '');
html = html.replace(/<script[^>]*><\/script>/g, '');

// Write bundled file
fs.writeFileSync('bundle.html', html);

// Clean up dist folder
const distPath = path.join(__dirname, 'dist');
if (fs.existsSync(distPath)) {
  const rimraf = require('rimraf');
  rimraf.sync(distPath);
}

console.log('✅ Bundle created: bundle.html');
console.log('📝 You can now share bundle.html as an artifact!');