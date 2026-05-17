# Copyright (c) Meta Platforms, Inc. and affiliates.


import torch

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 3 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
from PIL import Image
from sam3.visualization_utils import show_box, show_mask, show_points

from _paths import BPE_PATH, SAM3_CHECKPOINT, SAM3_ROOT

sam3_root = SAM3_ROOT

# font size for axes titles
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["figure.titlesize"] = 12


from sam3.model_builder import build_sam3_video_model

sam3_model = build_sam3_video_model(
    checkpoint_path=SAM3_CHECKPOINT, load_from_HF=False, bpe_path=BPE_PATH
)
predictor = sam3_model.tracker
predictor.backbone = sam3_model.detector.backbone


video_path = f"{sam3_root}/assets/videos/bedroom.mp4"
inference_state = predictor.init_state(video_path=video_path)


predictor.clear_all_points_in_video(inference_state)


# load the frames for visualization
cap = cv2.VideoCapture(video_path)
video_frames_for_vis = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()
frame0 = video_frames_for_vis[0]

width, height = frame0.shape[1], frame0.shape[0]


ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Let's add a positive click at (x, y) = (210, 350) to get started
points = np.array([[210, 350]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)

rel_points = [[x / width, y / height] for x, y in points]

points_tensor = torch.tensor(rel_points, dtype=torch.float32)
points_labels_tensor = torch.tensor(labels, dtype=torch.int32)

_, out_obj_ids, low_res_masks, video_res_masks = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points_tensor,
    labels=points_labels_tensor,
    clear_old_points=False,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(frame0)
show_points(points, labels, plt.gca())
show_mask((video_res_masks[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])


ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
# sending all clicks (and their labels) to `add_new_points_or_box`
points = np.array([[210, 350], [250, 220]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1], np.int32)

rel_points = [[x / width, y / height] for x, y in points]

points_tensor = torch.tensor(rel_points, dtype=torch.float32)
points_labels_tensor = torch.tensor(labels, dtype=torch.int32)

_, out_obj_ids, low_res_masks, video_res_masks  = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points_tensor,
    labels=points_labels_tensor,
    clear_old_points=False,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(frame0)
show_points(points, labels, plt.gca())
show_mask((video_res_masks[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])


# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(inference_state, start_frame_idx=0, max_frame_num_to_track=240, reverse=False, propagate_preflight=True):
    video_segments[frame_idx] = {
        out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 30
plt.close("all")
for out_frame_idx in range(0, len(video_frames_for_vis), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(video_frames_for_vis[out_frame_idx])
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)


ann_frame_idx = 150  # further refine some details on this frame
ann_obj_id = 1  # give a unique id to the object we interact with (it can be any integers)

# show the segment before further refinement
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx} -- before refinement")
plt.imshow(video_frames_for_vis[ann_frame_idx])
show_mask(video_segments[ann_frame_idx][ann_obj_id], plt.gca(), obj_id=ann_obj_id)

# Let's add a negative click on this frame at (x, y) = (82, 415) to refine the segment
points = np.array([[82, 410]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([0], np.int32)

rel_points = [[x / width, y / height] for x, y in points]

points_tensor = torch.tensor(rel_points, dtype=torch.float32)
points_labels_tensor = torch.tensor(labels, dtype=torch.int32)

_, out_obj_ids, low_res_masks, video_res_masks  = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points_tensor,
    labels=points_labels_tensor,
    clear_old_points=False,
)


# show the segment after the further refinement
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx} -- after refinement")
plt.imshow(video_frames_for_vis[ann_frame_idx])
show_points(points, labels, plt.gca())
show_mask((video_res_masks > 0.0).cpu().numpy(), plt.gca(), obj_id=ann_obj_id)


# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(inference_state, start_frame_idx=0, max_frame_num_to_track=300, reverse=False, propagate_preflight=True):
    video_segments[frame_idx] = {
        out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 30
plt.close("all")
for out_frame_idx in range(0, len(video_frames_for_vis), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(video_frames_for_vis[out_frame_idx])
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)


predictor.clear_all_points_in_video(inference_state)


ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)

# Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
box = np.array([[300, 0, 500, 400]], dtype=np.float32)

rel_box = [[xmin / width, ymin / height, xmax / width, ymax / height] for xmin, ymin, xmax, ymax in box]
rel_box = np.array(rel_box, dtype=np.float32)

_, out_obj_ids, low_res_masks, video_res_masks  = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    box=rel_box,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(video_frames_for_vis[ann_frame_idx])
show_box(box[0], plt.gca())
show_mask((video_res_masks[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=ann_obj_id)


ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)

# Let's add a positive click at (x, y) = (460, 60) to refine the mask
points = np.array([[460, 60]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
# note that we also need to send the original box input along with
# the new refinement click together into `add_new_points_or_box`
box = np.array([[300, 0, 500, 400]], dtype=np.float32)

rel_box = [[xmin / width, ymin / height, xmax / width, ymax / height] for xmin, ymin, xmax, ymax in box]
rel_box = np.array(rel_box, dtype=np.float32)

rel_points = [[x / width, y / height] for x, y in points]

points_tensor = torch.tensor(rel_points, dtype=torch.float32)
points_labels_tensor = torch.tensor(labels, dtype=torch.int32)

_, out_obj_ids, low_res_masks, video_res_masks  = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points_tensor,
    labels=points_labels_tensor,
    box=rel_box,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(video_frames_for_vis[ann_frame_idx])
show_box(box[0], plt.gca())
show_points(points, labels, plt.gca())
show_mask((video_res_masks[0][0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])


# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(inference_state, start_frame_idx=0, max_frame_num_to_track=300, reverse=False, propagate_preflight=True):
    video_segments[frame_idx] = {
        out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 30
plt.close("all")
for out_frame_idx in range(0, len(video_frames_for_vis), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(video_frames_for_vis[out_frame_idx])
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)


predictor.clear_all_points_in_video(inference_state)


prompts = {}  # hold all the clicks we add for visualization


ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

# Let's add a positive click at (x, y) = (200, 300) to get started on the first object
points = np.array([[200, 300]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels

rel_points = [[x / width, y / height] for x, y in points]
points_tensor = torch.tensor(rel_points, dtype=torch.float32)
points_labels_tensor = torch.tensor(labels, dtype=torch.int32)

_, out_obj_ids, low_res_masks, video_res_masks = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points_tensor,
    labels=points_labels_tensor,
)


# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(video_frames_for_vis[ann_frame_idx])
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(points, labels, plt.gca())
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((video_res_masks[i][0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)


# add the first object
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

# Let's add a 2nd negative click at (x, y) = (275, 175) to refine the first object
# sending all clicks (and their labels) to `add_new_points_or_box`
points = np.array([[200, 300], [275, 175]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 0], np.int32)
prompts[ann_obj_id] = points, labels

rel_points = [[x / width, y / height] for x, y in points]
points_tensor = torch.tensor(rel_points, dtype=torch.float32)
points_labels_tensor = torch.tensor(labels, dtype=torch.int32)


_, out_obj_ids, low_res_masks, video_res_masks  = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=rel_points,
    labels=points_labels_tensor,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(video_frames_for_vis[ann_frame_idx])
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(points, labels, plt.gca())
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((video_res_masks[i][0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)


ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)

# Let's now move on to the second object we want to track (giving it object id `3`)
# with a positive click at (x, y) = (400, 150)
points = np.array([[400, 150]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels

rel_points = [[x / width, y / height] for x, y in points]
points_tensor = torch.tensor(rel_points, dtype=torch.float32)
points_labels_tensor = torch.tensor(labels, dtype=torch.int32)


# `add_new_points_or_box` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, low_res_masks, video_res_masks = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points_tensor,
    labels=points_labels_tensor,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(video_frames_for_vis[ann_frame_idx])
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(points, labels, plt.gca())
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((video_res_masks[i][0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)


# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(inference_state, start_frame_idx=0, max_frame_num_to_track=300, reverse=False, propagate_preflight=True):
    video_segments[frame_idx] = {
        out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 30
plt.close("all")
for out_frame_idx in range(0, len(video_frames_for_vis), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(video_frames_for_vis[out_frame_idx])
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
