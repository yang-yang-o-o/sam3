# Copyright (c) Meta Platforms, Inc. and affiliates.


import os

import torch

from _paths import SAM3_CHECKPOINT, SAM3_ROOT

sam3_root = SAM3_ROOT

# Output directory for videos and preview images (works on headless cloud servers)
OUTPUT_DIR = os.environ.get(
    "SAM3_VIDEO_OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "output")
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
VIDEO_FPS = int(os.environ.get("SAM3_VIDEO_FPS", "24"))

# use all available GPUs on the machine
gpus_to_use = range(torch.cuda.device_count())
# # use only a single GPU
# gpus_to_use = [torch.cuda.current_device()]


from sam3.model_builder import build_sam3_video_predictor

predictor = build_sam3_video_predictor(
    gpus_to_use=gpus_to_use, checkpoint_path=SAM3_CHECKPOINT
)


import glob

import cv2
import numpy as np
from PIL import Image
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    render_formatted_masks_on_frame,
    save_formatted_mask_video,
)


def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame


def save_propagation_results(
    name,
    video_frames,
    outputs_per_frame,
    sample_frame_stride=60,
):
    """Save full tracking video and sample preview PNGs to OUTPUT_DIR."""
    video_path = os.path.join(OUTPUT_DIR, f"{name}.mp4")
    save_formatted_mask_video(
        video_frames,
        outputs_per_frame,
        video_path,
        fps=VIDEO_FPS,
    )

    preview_dir = os.path.join(OUTPUT_DIR, f"{name}_frames")
    os.makedirs(preview_dir, exist_ok=True)
    frame_indices = sorted(outputs_per_frame.keys())
    for frame_idx in range(0, len(frame_indices), sample_frame_stride):
        if frame_idx >= len(frame_indices):
            break
        actual_idx = frame_indices[frame_idx]
        img = load_frame(video_frames[actual_idx])
        overlay = render_formatted_masks_on_frame(
            img, outputs_per_frame[actual_idx]
        )
        Image.fromarray(overlay).save(
            os.path.join(preview_dir, f"frame_{actual_idx:05d}.png")
        )

    print(f"Saved results for '{name}' under {OUTPUT_DIR}")


def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")


# "video_path" needs to be either a JPEG folder or a MP4 video file
video_path = f"{sam3_root}/assets/videos/0001"


# load "video_frames_for_vis" for visualization purposes (they are not used by the model)
if isinstance(video_path, str) and video_path.endswith(".mp4"):
    cap = cv2.VideoCapture(video_path)
    video_frames_for_vis = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
else:
    video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
    try:
        # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
        video_frames_for_vis.sort(
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )
    except ValueError:
        # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
        print(
            f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
            f"falling back to lexicographic sort."
        )
        video_frames_for_vis.sort()


response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
session_id = response["session_id"]


# note: in case you already ran one text prompt and now want to switch to another text prompt
# it's required to reset the session first (otherwise the results would be wrong)
_ = predictor.handle_request(
    request=dict(
        type="reset_session",
        session_id=session_id,
    )
)


prompt_text_str = "person"
frame_idx = 0  # add a text prompt on frame 0
response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_idx,
        text=prompt_text_str,
    )
)
out = response["outputs"]

frame0_vis = prepare_masks_for_visualization({frame_idx: out})
Image.fromarray(
    render_formatted_masks_on_frame(
        load_frame(video_frames_for_vis[frame_idx]), frame0_vis[frame_idx]
    )
).save(os.path.join(OUTPUT_DIR, "00_text_prompt_frame0.png"))


# now we propagate the outputs from frame 0 to the end of the video and collect all outputs
outputs_per_frame = propagate_in_video(predictor, session_id)

# finally, we reformat the outputs for visualization and save video + preview frames
outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)
save_propagation_results("01_text_person", video_frames_for_vis, outputs_per_frame)


# we pick id 2, which is the dancer in the front
obj_id = 2
response = predictor.handle_request(
    request=dict(
        type="remove_object",
        session_id=session_id,
        obj_id=obj_id,
    )
)


# now we propagate the outputs from frame 0 to the end of the video and collect all outputs
outputs_per_frame = propagate_in_video(predictor, session_id)

outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)
save_propagation_results(
    "02_after_remove_obj2", video_frames_for_vis, outputs_per_frame
)


sample_img = Image.fromarray(load_frame(video_frames_for_vis[0]))

IMG_WIDTH, IMG_HEIGHT = sample_img.size


# let's add back the dancer via point prompts.
# we will use a single positive click to add the dancer back.

frame_idx = 0
obj_id = 2
points_abs = np.array(
    [
        [760, 550],  # positive click
    ]
)
# positive clicks have label 1, while negative clicks have label 0
labels = np.array([1])


# convert points and labels to tensors; also convert to relative coordinates
points_tensor = torch.tensor(
    abs_to_rel_coords(points_abs, IMG_WIDTH, IMG_HEIGHT, coord_type="point"),
    dtype=torch.float32,
)
points_labels_tensor = torch.tensor(labels, dtype=torch.int32)

response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_idx,
        points=points_tensor,
        point_labels=points_labels_tensor,
        obj_id=obj_id,
    )
)
out = response["outputs"]

frame0_vis = prepare_masks_for_visualization({frame_idx: out})
Image.fromarray(
    render_formatted_masks_on_frame(
        load_frame(video_frames_for_vis[frame_idx]), frame0_vis[frame_idx]
    )
).save(os.path.join(OUTPUT_DIR, "03_point_prompt_frame0.png"))


# now we propagate the outputs from frame 0 to the end of the video and collect all outputs
outputs_per_frame = propagate_in_video(predictor, session_id)

outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)
save_propagation_results(
    "03_point_prompt_dancer", video_frames_for_vis, outputs_per_frame
)


# For the dancer in the front, suppose now we only want to segment her T-shirt instead of her whole body
# we will use 2 positive clicks and 2 negative clicks to select her shirt.

frame_idx = 0
obj_id = 2
points_abs = np.array(
    [
        [740, 450],  # positive click
        [760, 630],  # negative click
        [840, 640],  # negative click
        [760, 550],  # positive click
    ]
)
# positive clicks have label 1, while negative clicks have label 0
labels = np.array([1, 0, 0, 1])


# convert points and labels to tensors; also convert to relative coordinates
points_tensor = torch.tensor(
    abs_to_rel_coords(points_abs, IMG_WIDTH, IMG_HEIGHT, coord_type="point"),
    dtype=torch.float32,
)
points_labels_tensor = torch.tensor(labels, dtype=torch.int32)

response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_idx,
        points=points_tensor,
        point_labels=points_labels_tensor,
        obj_id=obj_id,
    )
)
out = response["outputs"]

frame0_vis = prepare_masks_for_visualization({frame_idx: out})
Image.fromarray(
    render_formatted_masks_on_frame(
        load_frame(video_frames_for_vis[frame_idx]), frame0_vis[frame_idx]
    )
).save(os.path.join(OUTPUT_DIR, "04_tshirt_prompt_frame0.png"))


# now we propagate the outputs from frame 0 to the end of the video and collect all outputs
outputs_per_frame = propagate_in_video(predictor, session_id)

outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)
save_propagation_results(
    "04_tshirt_refinement", video_frames_for_vis, outputs_per_frame
)


# finally, close the inference session to free its GPU resources
# (you may start a new session on another video)
_ = predictor.handle_request(
    request=dict(
        type="close_session",
        session_id=session_id,
    )
)


# after all inference is done, we can shutdown the predictor
# to free up the multi-GPU process group
predictor.shutdown()

print(f"\nAll outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
print("  Videos: 01_text_person.mp4, 02_after_remove_obj2.mp4, ...")
print("  Preview frames: <name>_frames/frame_*.png")
