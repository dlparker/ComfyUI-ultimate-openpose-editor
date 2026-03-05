# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A ComfyUI custom node extension for editing OpenPose keypoints interactively. It provides:
- **OpenposeEditorNode**: A node with an iframe-based visual editor (opened via right-click menu) for editing pose keypoints before they flow through the graph.
- **AppendageEditorNode**: A procedural node for programmatically transforming specific body parts (arms, legs, hands, feet, torso) with scale/rotate/offset controls.

## Installation Context

This lives in ComfyUI's `custom_nodes/` directory. There is no build step — Python files are loaded directly by ComfyUI, and JS files are served from the `js/` directory (registered via `WEB_DIRECTORY = "js"` in `__init__.py`).

To install dependencies:
```bash
pip install -r requirements.txt
```

There are no test suites or linting configs in this project.

## Architecture

### Data Flow

Pose data is stored as a list of image objects (JSON):
```json
[{
  "canvas_width": 512,
  "canvas_height": 768,
  "people": [{
    "pose_keypoints_2d": [...],      // body: 18 keypoints, flat [x,y,conf, x,y,conf, ...]
    "face_keypoints_2d": [...],      // face: 70 keypoints
    "hand_left_keypoints_2d": [...], // 21 keypoints each
    "hand_right_keypoints_2d": [...]
  }]
}]
```

Keypoints exist in two coordinate spaces:
- **Pixel space**: x/y values in the hundreds (e.g., 109–512 range), used when data comes in from external pose detectors
- **Normalized space**: x/y values in [0.0, 1.0], used internally for rendering

### Normalization (`util.py: pose_normalized`)

`pose_normalized()` detects which space the data is in by checking `max(keypoints) > 2.0`. If pixel space is detected, it divides all x values by `canvas_width` and y values by `canvas_height`. **This is the source of the known bug** (see below).

### Rendering Pipeline (`util.py`)

1. `pose_normalized()` — converts pixel-space keypoints to normalized [0,1] space
2. `draw_pose_json()` — applies scale transforms, builds candidate/subset arrays for body
3. `draw_pose()` → `draw_bodypose()` / `draw_handpose()` / `draw_facepose()` — renders to numpy canvas using OpenCV

Body keypoints are stored in a `candidate` list (each entry is `[norm_x, norm_y]`) and a `subset` array (indices into candidate). During rendering, coordinates are multiplied back by `W` and `H` to get pixel positions.

### JS/Frontend (`js/`)

- `openpose_editor.js` — Registers the ComfyUI extension, adds the "Open in Openpose Editor" right-click menu item, manages an `OpenposeEditorDialog` (singleton iframe dialog). Communicates with the iframe via `postMessage`.
- `editor_textDisplay.js` — Registers a separate extension that displays the output POSE_JSON as a read-only text widget on the node after execution.
- `js/ui/OpenposeEditor.html` + `js/ui/assets/index-cd1939ba.js` — The compiled Vue/React editor app running inside the iframe. This is a pre-built bundle (not source).

### Communication Protocol (ComfyUI node ↔ iframe)

- **Node → iframe**: `postMessage({ modalId: 0, poses: [...] })` to load pose data into editor
- **iframe → node**: `postMessage({ modalId: 0, poses: [...] })` to send edited poses back; the JS writes the result into `targetNode.widgets[14].element` (the POSE_JSON textarea widget)
- **iframe → node**: `postMessage({ ready: true })` when the iframe is loaded

### Python Node Entry Points

- `__init__.py` — Registers both nodes with ComfyUI
- `openpose_editor_nodes.py` — `OpenposeEditorNode.load_pose()`: orchestrates normalization, scaling, and rendering
- `appendage_editor_nodes.py` — `AppendageEditorNode`: procedural body-part transforms
- `util.py` — All rendering and math logic

## Known Bug: Body Keypoints Displaying in Wrong Position

**Symptom**: Body keypoints (shoulders, elbows, wrists, etc.) appear clustered in the upper-left corner of the editor canvas, while face and hand keypoints render correctly.

**Root Cause**: The normalization detection in `pose_normalized()` (`util.py:111`) uses `max(body) > 2.0` to detect pixel-space data. However, the check tests each keypoint array independently. If body keypoints happen to all be in the range ~109–110 (which is > 2.0, so they should be normalized), but face/hand data is already normalized (< 2.0), then only the body data gets divided — or the detection fires based on face/hand data and body data is then divided again, or vice versa. The logic scans arrays sequentially and `break`s on first `> 2.0` match, meaning it correctly identifies the coordinate space but the issue may be in how the editor (the iframe bundle) interprets the data vs. how `util.py` processes it.

The editor canvas in the iframe uses the raw pose JSON data passed via `postMessage` in `openpose_editor.js:setCanvasJSONString()`. The iframe bundle likely expects normalized [0,1] coordinates. If the data coming in is in pixel space and the iframe scales differently than the Python renderer, body points (with tight x-value ranges like 109–110) would appear compressed in the corner.

The fix likely involves ensuring consistent coordinate space when passing data from the node to the iframe editor, or fixing the normalization logic to handle the case where body data is in a different space than face/hand data.
