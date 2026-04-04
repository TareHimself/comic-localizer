# Extension Frontend

This is a basic extension example

## building

- You need nodejs to build this extension
```bash
npm install
```
- To build a Chromium package run
```bash
npm run build:chromium
```
- To build a Firefox package run
```bash
npm run build:firefox
```

- Build outputs are written to:
	- `dist/chromium`
	- `dist/firefox`
- Common manifest fields are edited in `public/manifest.json`. Build-time browser-specific fields are added automatically.
