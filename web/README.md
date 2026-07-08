# Web Frontend

Browser-based 3D visualization for the robotic arm, built with Three.js and the URDF loader.

## Overview

This directory will contain the frontend code migrated from the original `playground/` project.

### Features

- Load and render the robotic arm URDF model in the browser
- Visualize trajectory playback (pick-place-pour sequences)
- Real-time joint angle display
- Vision overlay (camera feed with detection results)

## Structure (planned)

```
web/
├── package.json
├── vite.config.js
├── public/
│   ├── URDF/genkiarm.urdf
│   └── trajectories/       # Selected trajectory JSON files
└── src/
    ├── index.js
    ├── index_v8.js
    ├── trajectoryPlayback.js
    ├── robotConfig.js
    └── robotControls.js
```

## Notes

- `node_modules/` and `dist/` are excluded by `.gitignore`
- Run `npm install` (or `pnpm install`) after cloning
- Run `npm run dev` to start the dev server
