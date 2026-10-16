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

- Execution API contract: `/api/assistant/execute` requires a top-level JSON boolean
  `confirm: true`, set only by a separate operator confirmation action. Task-level
  `confirm_before_run` is a required policy flag, not execution consent.
- `/api/assistant/plan` always remains planning-only, including when given confirmation fields.
- Frontend code is not present here yet; its confirmation UI has not been verified.
  See [执行授权说明](../docs/execution_authorization_cn.md) for the contract and remaining boundaries.

- `node_modules/` and `dist/` are excluded by `.gitignore`
- Run `npm install` (or `pnpm install`) after cloning
- Run `npm run dev` to start the dev server
