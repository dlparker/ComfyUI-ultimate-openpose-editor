import json
import os
import hashlib
import datetime
import torch
import numpy as np
from server import PromptServer
from .util import draw_pose_json, draw_pose, extend_scalelist, pose_normalized

OpenposeJSON = dict

# ---------- debug logging ----------
DEBUG_LOG_DIR = "Y:/comfyui_logs"

def _pose_array_stats(arr, label):
    if not arr:
        return f"    {label}: empty"
    confs = [arr[i] for i in range(2, len(arr), 3)]
    x_vals = [arr[i] for i in range(0, len(arr), 3)]
    y_vals = [arr[i] for i in range(1, len(arr), 3)]
    sample = arr[:9]  # first 3 keypoints
    return (
        f"    {label}: {len(arr)//3} keypoints | "
        f"x=[{min(x_vals):.4f}, {max(x_vals):.4f}] "
        f"y=[{min(y_vals):.4f}, {max(y_vals):.4f}] "
        f"conf=[{min(confs):.2f}, {max(confs):.2f}] | "
        f"first 3 kpts: {[round(v,4) for v in sample]}"
    )

def _log_pose_json(f, stage, pose_json_str):
    try:
        data = json.loads(pose_json_str) if isinstance(pose_json_str, str) else pose_json_str
        if isinstance(data, dict):
            data = [data]
        f.write(f"\n=== {stage} ===\n")
        for img_i, image in enumerate(data):
            f.write(f"  image[{img_i}]: canvas={image.get('canvas_width')}x{image.get('canvas_height')}\n")
            for person_i, person in enumerate(image.get('people', [])):
                f.write(f"  person[{person_i}]:\n")
                for key in ['pose_keypoints_2d', 'face_keypoints_2d',
                            'hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
                    arr = person.get(key)
                    if arr:
                        f.write(_pose_array_stats(arr, key) + "\n")
                    else:
                        f.write(f"    {key}: absent/empty\n")
    except Exception as e:
        f.write(f"  [error logging stage '{stage}': {e}]\n")

def _open_debug_log():
    try:
        os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(DEBUG_LOG_DIR, f"openpose_debug_{ts}.log")
        return open(path, "w", encoding="utf-8")
    except Exception as e:
        print(f"[OpenposeEditor] Could not open debug log: {e}")
        return None
# ---------- end debug logging ----------

class OpenposeEditorNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "show_body": ("BOOLEAN", {"default": True}),
                "show_face": ("BOOLEAN", {"default": True}),
                "show_hands": ("BOOLEAN", {"default": True}),
                "resolution_x": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 12800,
                    "tooltip": "Resolution X. -1 means use the original resolution."
                }),
                "pose_marker_size": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 100
                }),
                "face_marker_size": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 100
                }),
                "hand_marker_size": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 100
                }),
                "hands_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05
                }),
                "body_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05
                }),
                "head_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05
                }),
                "overall_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05
                }),
                "scalelist_behavior": (["poses", "images"], {"default": "poses", "tooltip": "When the scale input is a list, this determines how the scale list takes effect, the differences appear when there are multiple persons(poses) in one image."}),
                "match_scalelist_method": (["no extend", "loop extend", "clamp extend"], {"default": "loop extend", "tooltip": "Match the scale list to the input poses or images when the scale list length is shorter. No extend: Beyound the scale list will be 1.0. Loop: Loop the scale list to match the poses or images length. Clamp: Use the last scale value to extend the scale list."}),
                "only_scale_pose_index": ("INT", {
                    "default": 99,
                    "min": -100,
                    "max": 100,
                    "tooltip": "For multiple poses in one image, the scale will be only applied at desired index. If set to a number larger than the number of poses in the image, the scale will be applied to all poses. Negative number will apply to the pose from the end."
                }),
                "POSE_JSON": ("STRING", {"multiline": True}),
                "POSE_KEYPOINT": ("POSE_KEYPOINT",{"default": None}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_NAMES = ("POSE_IMAGE", "POSE_KEYPOINT", "POSE_JSON")
    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT", "STRING")
    OUTPUT_NODE = True
    FUNCTION = "load_pose"
    CATEGORY = "ultimate-openpose"

    def load_pose(self, show_body, show_face, show_hands, resolution_x, pose_marker_size, face_marker_size, hand_marker_size, hands_scale, body_scale, head_scale, overall_scale, scalelist_behavior, match_scalelist_method, only_scale_pose_index, POSE_JSON: str, POSE_KEYPOINT=None, unique_id=None) -> tuple[OpenposeJSON]:
        '''
        priority output is: POSE_JSON > POSE_KEYPOINT
        priority edit is: POSE_JSON > POSE_KEYPOINT

        When POSE_JSON is non-empty (e.g. written back by the editor after
        the user makes edits), it takes precedence over a connected
        POSE_KEYPOINT. This means edits take effect immediately on the next
        run without having to disconnect the POSE_KEYPOINT wire.

        To revert to the live POSE_KEYPOINT (e.g. after changing the source
        image), clear the POSE_JSON textarea.
        '''
        # Invalidate edited POSE_JSON if the upstream POSE_KEYPOINT has changed
        # (e.g. user switched to a different source image). Instance variable
        # persists across runs since ComfyUI keeps node instances alive.
        if POSE_KEYPOINT is not None:
            kp_hash = hashlib.md5(
                json.dumps(POSE_KEYPOINT, sort_keys=True, default=str).encode()
            ).hexdigest()
            prev_hash = getattr(self, '_last_keypoint_hash', None)
            invalidated_hash = getattr(self, '_invalidated_for_hash', None)

            if POSE_JSON:
                if prev_hash is not None and kp_hash != prev_hash:
                    # New upstream pose — first detection, notify user
                    POSE_JSON = ''
                    self._invalidated_for_hash = kp_hash
                    PromptServer.instance.send_sync("openpose_editor_event", {
                        "type": "keypoint_invalidated",
                        "node_id": unique_id,
                        "show_toast": True,
                    })
                elif invalidated_hash is not None and kp_hash == invalidated_hash:
                    # Widget wasn't cleared yet — keep suppressing, nudge JS again
                    POSE_JSON = ''
                    PromptServer.instance.send_sync("openpose_editor_event", {
                        "type": "keypoint_invalidated",
                        "node_id": unique_id,
                        "show_toast": False,
                    })
            elif invalidated_hash is not None:
                # Widget is now empty — invalidation complete, reset flag
                self._invalidated_for_hash = None

            self._last_keypoint_hash = kp_hash

        dbg = _open_debug_log()
        try:
          if POSE_JSON:
            POSE_JSON = POSE_JSON.replace("'",'"').replace('None','[]')
            POSE_PASS = POSE_JSON

            if dbg:
                dbg.write(f"code path: POSE_JSON present (POSE_KEYPOINT={'present' if POSE_KEYPOINT is not None else 'absent'})\n")
                _log_pose_json(dbg, "INPUT (POSE_PASS before pose_normalized)", POSE_PASS)

            # parse the JSON
            hands_scalelist, body_scalelist, head_scalelist, overall_scalelist = extend_scalelist(
                scalelist_behavior, POSE_PASS, hands_scale, body_scale, head_scale, overall_scale,
                match_scalelist_method, only_scale_pose_index)
            normalized_pose_json = pose_normalized(POSE_PASS)

            if dbg:
                _log_pose_json(dbg, "AFTER pose_normalized", normalized_pose_json)

            pose_imgs, POSE_PASS_SCALED = draw_pose_json(normalized_pose_json, resolution_x, show_body, show_face, show_hands, pose_marker_size, face_marker_size, hand_marker_size, hands_scalelist, body_scalelist, head_scalelist, overall_scalelist)

            if dbg:
                _log_pose_json(dbg, "AFTER draw_pose_json (output/POSE_PASS_SCALED)", POSE_PASS_SCALED)

            if pose_imgs:
                pose_imgs_np = np.array(pose_imgs).astype(np.float32) / 255
                return {
                    "ui": {"POSE_JSON": [json.dumps(POSE_PASS_SCALED, indent=4)]},
                    "result": (torch.from_numpy(pose_imgs_np), POSE_PASS_SCALED, json.dumps(POSE_PASS_SCALED,indent=4))
                }
          elif POSE_KEYPOINT is not None:
            POSE_JSON = json.dumps(POSE_KEYPOINT,indent=4).replace("'",'"').replace('None','[]')

            if dbg:
                dbg.write("code path: POSE_KEYPOINT only (no POSE_JSON)\n")
                _log_pose_json(dbg, "INPUT (POSE_KEYPOINT before pose_normalized)", POSE_JSON)

            hands_scalelist, body_scalelist, head_scalelist, overall_scalelist = extend_scalelist(
                scalelist_behavior, POSE_JSON, hands_scale, body_scale, head_scale, overall_scale,
                match_scalelist_method, only_scale_pose_index)
            normalized_pose_json = pose_normalized(POSE_JSON)

            if dbg:
                _log_pose_json(dbg, "AFTER pose_normalized", normalized_pose_json)

            pose_imgs, POSE_SCALED = draw_pose_json(normalized_pose_json, resolution_x, show_body, show_face, show_hands, pose_marker_size, face_marker_size, hand_marker_size, hands_scalelist, body_scalelist, head_scalelist, overall_scalelist)

            if dbg:
                _log_pose_json(dbg, "AFTER draw_pose_json (output/POSE_SCALED)", POSE_SCALED)

            if pose_imgs:
                pose_imgs_np = np.array(pose_imgs).astype(np.float32) / 255
                return {
                    "ui": {"POSE_JSON": [json.dumps(POSE_SCALED, indent=4)]},
                    "result": (torch.from_numpy(pose_imgs_np), POSE_SCALED, json.dumps(POSE_SCALED, indent=4))
                }
        finally:
          if dbg:
            dbg.close()

        # otherwise output blank images
        W=512
        H=768
        pose_draw = dict(bodies={'candidate':[], 'subset':[]}, faces=[], hands=[])
        pose_out = dict(pose_keypoints_2d=[], face_keypoints_2d=[], hand_left_keypoints_2d=[], hand_right_keypoints_2d=[])
        people=[dict(people=[pose_out], canvas_height=H, canvas_width=W)]

        W_scaled = resolution_x
        if resolution_x < 64:
            W_scaled = W
        H_scaled = int(H*(W_scaled*1.0/W))
        pose_img = [draw_pose(pose_draw, H_scaled, W_scaled, pose_marker_size, face_marker_size, hand_marker_size)]
        pose_img_np = np.array(pose_img).astype(np.float32) / 255

        return {
                "ui": {"POSE_JSON": people},
                "result": (torch.from_numpy(pose_img_np), people, json.dumps(people))
        }
