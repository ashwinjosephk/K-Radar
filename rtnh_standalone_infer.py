"""
Standalone RTNH inference / visualization script for the K-Radar devkit.

WHY THIS SCRIPT EXISTS
-----------------------
The devkit's own main_test_0.py runs everything through PipelineDetection_v1_0,
which also builds an optimizer, a scheduler, a tensorboard logger, and KITTI-eval
machinery -- none of which you need just to load a checkpoint and look at what
it predicts. This script rebuilds only the three things RTNH actually needs at
inference time (config, dataset, network) by calling the exact same devkit
functions the pipeline calls internally. That keeps it small enough to read
top to bottom and easy to step through with a debugger.

Relevant devkit files, if you want to compare against the original:
    pipelines/pipeline_detection_v1_0.py -> load_dict_model()   (loading weights)
    utils/util_pipeline.py               -> build_network(), build_dataset()
    models/skeletons/rdr_base.py         -> RadarBase.forward()  (module chain)
    models/backbone_3d/rdr_sp_pw.py      -> RadarSparseBackbone (writes 'bev_feat')

WHAT "RadarBase.forward()" ACTUALLY DOES
------------------------------------------
The whole network is just four steps chained together:
    for module in [pre_processor, backbone, head, roi_head]:
        x = module(x)
So you can stop that chain early (e.g. right after `backbone`) to pull out the
BEV feature map before the detection head ever runs. That's useful if you want
RTNH's radar features as an input to something else downstream (a world model,
a VLA-style planner, etc.) rather than just the final boxes.

MODES -- pick the flags for what you want
-------------------------------------------
1) Feature extraction (stop before the detection head):
     --extract-bev-feat              dump dict_item['bev_feat'] to a .pt file
     --extract-bev-feat --visualize  also save a heatmap PNG of it

2) Full detection + look at the results (these can be combined in one run):
     --visualize-detections   top-down (BEV) point-cloud + box plot -> PNG
     --project-cam            boxes projected onto a camera photo    -> PNG
     --open3d-interactive     a real, rotatable 3D window (needs a display)
     --combined               BEV + camera panels side by side in one PNG

3) Neither of the above: just runs inference once and prints the output
   dict's keys, so you can see what a checkpoint actually returns.

Run from inside the K-Radar repo so the devkit's own imports resolve:
    cd K-Radar
    python rtnh_standalone_infer.py \
        --cfg configs/cfg_RTNH_wide.yml \
        --ckpt pretrained/RTNH_wide_10.pt \
        --idx 0 --combined
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset


# ============================================================================
# Setup: load the config, dataset, and network -- no Pipeline wrapper needed
# ============================================================================

def build_from_scratch(path_cfg: str, path_ckpt: str, device: str = "cuda"):
    """Rebuilds cfg -> dataset -> network -> loaded weights, no Pipeline wrapper."""
    from utils.util_config import cfg, cfg_from_yaml_file
    from models.skeletons import build_skeleton
    import datasets

    cfg = cfg_from_yaml_file(path_cfg, cfg)

    # We only ever need the test split here -- this script is for inference.
    dataset_test = datasets.__all__[cfg.DATASET.NAME](cfg=cfg, split="test")

    # Same call the pipeline makes internally (utils/util_pipeline.py:build_network).
    network = build_skeleton(cfg).to(device)

    # Same call as PipelineDetection_v1_0.load_dict_model() -- it's a plain
    # state_dict, no 'model_state_dict' wrapper, no optimizer/epoch bookkeeping.
    state_dict = torch.load(path_ckpt, map_location=device)
    missing, unexpected = network.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing (first 5):", missing[:5])
        if unexpected:
            print("  unexpected (first 5):", unexpected[:5])

    network.eval()
    return cfg, dataset_test, network


def resolve_camera_image(dict_item, cam_folder: str = "cam-front"):
    """
    Auto-locates the camera frame matching this sample.

    K-Radar stores a frame-index tuple in every label file's header
    ('rdr_ldr64_camf_ldr128_camr' -- see kradar_detection_v2_0.py's
    get_label()). dict_item['meta'][0]['idx']['camf'] is that exact camera
    frame number, so we don't have to ask the user to type a path by hand.

    The exact filename convention isn't exercised anywhere in this
    detection-only dataset class (it only ever loads radar/lidar/labels),
    so this just tries a few plausible patterns and reports what it tried
    if none of them exist.
    """
    meta = dict_item['meta'][0]
    header, seq, camf = meta['header'], meta['seq'], meta['idx']['camf']
    base_dir = Path(header) / seq / cam_folder

    candidates = [
        base_dir / f"{cam_folder}_{camf}.png",
        base_dir / f"{camf}.png",
        base_dir / f"{cam_folder}_{camf}.jpg",
        base_dir / f"{camf}.jpg",
        base_dir / f"{int(camf):05d}.png",
        base_dir / f"{int(camf):06d}.png",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    raise FileNotFoundError(
        f"Couldn't auto-locate the camera frame. seq={seq}, camf={camf}, "
        f"looked in {base_dir}. Tried: {[c.name for c in candidates]}. "
        f"Open that folder, check the real filename pattern, then pass it "
        f"explicitly with --cam-img."
    )


# ============================================================================
# Running the network
# ============================================================================

def run_full_forward(network, dict_item):
    """Standard path: pre_processor -> backbone -> head -> roi_head.
    In eval mode the head also runs NMS internally, so dict_item['pred_dicts']
    comes out already populated -- no extra postprocessing call needed."""
    with torch.no_grad():
        return network(dict_item)


def run_partial_forward(network, dict_item, stop_after="backbone"):
    """
    Manually replays RadarBase.forward()'s module loop but stops early.
    network.list_modules is always [pre_processor, backbone, head, roi_head]
    (see models/skeletons/rdr_base.py). Stopping after 'backbone' gets you
    dict_item['bev_feat'] -- shape (B, C, Y, X), C=768 for cfg_RTNH_wide.yml --
    before the detection head ever runs.
    """
    stop_names = [m for m in network.list_module_names if getattr(network, m, None) is not None]
    with torch.no_grad():
        for name, module in zip(stop_names, network.list_modules):
            dict_item = module(dict_item)
            if name == stop_after:
                break
    return dict_item


# ============================================================================
# Small shared helpers
# ============================================================================

def _get_class_names(dataset):
    """Builds the class-name list from dataset.label, same filtering
    main_test_0.py does (drop the non-class config keys, keep only classes
    actually used for detection)."""
    dict_label = dataset.label.copy()
    for k in ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']:
        dict_label.pop(k, None)
    class_names = []
    for k, v in dict_label.items():
        _, logit_idx, _, _ = v
        if logit_idx > 0:
            class_names.append(k)
    return class_names


def _bev_box_corners(x, y, l, w, theta):
    """The 4 corners of a box as seen from directly above (top-down / BEV),
    i.e. just the box's footprint on the ground. Same rotation convention as
    dataset.draw_3d_box_in_cylinder, just in 2D instead of 3D."""
    import numpy as np
    corners_local = np.array([[l / 2, w / 2], [l / 2, -w / 2],
                               [-l / 2, -w / 2], [-l / 2, w / 2]])
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    return corners_local @ rotation.T + np.array([x, y])


def _draw_legend_on_image(img_bgr, entries, org=(20, 40), line_gap=32):
    """
    Burns a small legend (colored line + label) directly into a cv2 BGR
    image, with a translucent background box behind it. entries is a list
    of (label_text, color_bgr) pairs.

    Used for the camera projection and (previously) the open3d screenshot --
    plain pixel images don't get a matplotlib-style legend() for free, so we
    draw one ourselves. Doing it this way also means the legend survives
    automatically when this image gets embedded into the --combined figure.
    """
    import cv2

    pad = 12
    longest_label_width = max(cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
                               for label, _ in entries)
    box_w = max(260, 40 + longest_label_width)
    box_h = pad * 2 + line_gap * len(entries)
    x0, y0 = org[0] - pad, org[1] - pad - 20

    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    img_bgr = cv2.addWeighted(overlay, 0.55, img_bgr, 0.45, 0)  # semi-transparent

    for i, (label, color) in enumerate(entries):
        y = org[1] + i * line_gap
        cv2.line(img_bgr, (org[0], y - 6), (org[0] + 30, y - 6), color, 4)
        cv2.putText(img_bgr, label, (org[0] + 40, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img_bgr


# ============================================================================
# Mode 1: feature extraction
# ============================================================================

def visualize_bev_feat(bev_feat: torch.Tensor, out_path: str):
    """
    bev_feat: (1, C, Y, X) tensor, straight from run_partial_forward(..., stop_after='backbone').
    Saves two heatmaps -- averaging across channels, and taking the max
    across channels -- as one PNG. Uses matplotlib's 'Agg' backend so this
    works over plain SSH with no display.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feat = bev_feat[0].detach().cpu()  # (C, Y, X)
    mean_map = feat.mean(dim=0).numpy()
    max_map = feat.max(dim=0).values.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, data, title in zip(axes, [mean_map, max_map], ["channel-mean", "channel-max"]):
        im = ax.imshow(data, origin="lower", cmap="viridis", aspect="auto")
        ax.set_title(f"bev_feat ({title})")
        ax.set_xlabel("X (range bins)")
        ax.set_ylabel("Y (cross-range bins)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"RTNH backbone BEV feature map -- shape {tuple(bev_feat.shape)}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {out_path}")


# ============================================================================
# Mode 2: detection + visualization
# ============================================================================

def print_detection_summary(dict_item, dataset, conf_thr: float = 0.5):
    """
    Prints every GT box, and EVERY predicted box (not just the ones above
    conf_thr), with class, distance, and score. Lets you tell whether a
    "missing" box is a genuine miss (no nearby prediction at all) or just a
    confidence-threshold cutoff (a nearby prediction exists but scored low).
    """
    class_names = _get_class_names(dataset)
    label = dict_item['label'][0]
    pred_dicts = dict_item['pred_dicts'][0]
    pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
    pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
    pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()

    print(f"\n=== GT boxes ({len(label)}) ===")
    gt_xy = []
    for i, (cls_name, _logit, (x, y, z, th, l, w, h), _trk) in enumerate(label):
        dist = (x**2 + y**2) ** 0.5
        gt_xy.append((x, y))
        print(f"  [{i}] {cls_name:12s} x={x:6.1f} y={y:6.1f} dist={dist:5.1f}m "
              f"l={l:.1f} w={w:.1f} h={h:.1f} th={th:+.2f}rad")

    order = pred_scores.argsort()[::-1]  # highest score first
    print(f"\n=== ALL predictions ({len(pred_scores)}), sorted by score, "
          f"threshold={conf_thr} ===")
    for i in order:
        x, y, z, l, w, h, th = pred_boxes[i]
        cls_idx = pred_labels[i]
        cls_name = class_names[cls_idx - 1] if 0 < cls_idx <= len(class_names) else f"cls{cls_idx}"
        dist = (x**2 + y**2) ** 0.5
        # distance to the nearest GT box center -- tells you if a low-score
        # prediction is at least near a real object, or off in empty space
        nearest_gt = min(((gx - x)**2 + (gy - y)**2)**0.5 for gx, gy in gt_xy) if gt_xy else float('nan')
        mark = "DRAWN" if pred_scores[i] > conf_thr else "     "
        print(f"  [{mark}] {cls_name:12s} score={pred_scores[i]:.3f} "
              f"x={x:6.1f} y={y:6.1f} dist={dist:5.1f}m nearest_GT={nearest_gt:5.1f}m")
    print()


def visualize_bev_detections(dict_item, dataset, out_path: str = None, conf_thr: float = 0.5, ax=None):
    """
    Top-down (BEV) plot: raw radar sparse points + GT boxes (green) +
    predicted boxes above conf_thr (red). This is the same information
    main_test_0.py shows in its open3d window, just as a flat 2D plot that
    saves to a file instead of needing a display.

    Pass an existing matplotlib Axes via `ax` to draw into a combined figure
    instead of creating and saving a standalone one.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.lines import Line2D

    pts = dict_item['rdr_sparse'].detach().cpu().numpy()  # (N, C>=3): x, y, z, ...

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(9, 9))

    ax.scatter(pts[:, 0], pts[:, 1], s=1.5, c='dimgray', alpha=0.5)

    for cls_name, _logit, (x, y, z, th, l, w, h), _trk in dict_item['label'][0]:
        corners = _bev_box_corners(x, y, l, w, th)
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor='limegreen', linewidth=2))

    pred_dicts = dict_item['pred_dicts'][0]
    pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
    pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
    n_drawn = 0
    for i in range(len(pred_scores)):
        if pred_scores[i] <= conf_thr:
            continue
        x, y, z, l, w, h, th = pred_boxes[i]
        corners = _bev_box_corners(x, y, l, w, th)
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor='red', linewidth=2))
        n_drawn += 1

    ax.set_xlabel("X [m] (forward)")
    ax.set_ylabel("Y [m] (left)")
    ax.set_aspect('equal')
    ax.set_title(f"BEV: {len(dict_item['label'][0])} GT (green) / {n_drawn} pred>{conf_thr} (red)")
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='none', markerfacecolor='dimgray',
               markeredgecolor='none', markersize=6, label='radar sparse pts'),
        Line2D([0], [0], color='limegreen', lw=2, label='GT box'),
        Line2D([0], [0], color='red', lw=2, label=f'Prediction (score>{conf_thr})'),
    ], loc='upper right')

    summary = f"{len(pts)} radar pts, {len(dict_item['label'][0])} GT boxes, {n_drawn} pred boxes drawn"
    if owns_fig:
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"saved -> {out_path}  ({summary})")
    else:
        print(f"BEV panel: {summary}")


def project_boxes_to_image(dict_item, dataset, image_path: str, out_path: str = None,
                            calib_yml: str = None, conf_thr: float = 0.5,
                            undistort: bool = True, return_array: bool = False):
    """
    Projects GT (green) + predicted (red, score>conf_thr) 3D boxes onto a
    camera image, using the devkit's own calibration math (utils/util_calib.py).

    IMPORTANT: K-Radar calibration is per-sequence, stored alongside your
    dataset download (e.g. an info_calib/ folder next to each sequence) --
    NOT in this git repo. Without --calib-yml pointing at the correct
    per-sequence file, this defaults to util_calib.dict_front0, which is
    only an illustrative example and can be off by a few cm/degrees for
    your actual sequence. Treat the default as "roughly right", not exact.

    Set return_array=True to get the annotated BGR image back in memory
    (e.g. to embed it into --combined) instead of, or as well as, writing it.
    """
    import cv2
    import numpy as np
    from utils.util_calib import get_matrices_from_dict_calib, dict_front0

    if calib_yml is not None:
        import yaml
        with open(calib_yml, 'r') as f:
            dict_calib = yaml.safe_load(f)
    else:
        print("[project_boxes_to_image] WARNING: no --calib-yml given, using "
              "util_calib.dict_front0 (illustrative default, not necessarily "
              "this sequence's real calibration)")
        dict_calib = dict_front0

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"cv2 could not read image: {image_path}")

    # Build the lidar-frame -> pixel-coordinate projection matrix.
    img_size, intrinsics, distortion, T_ldr2cam = get_matrices_from_dict_calib(dict_calib)
    if undistort:
        ncm, _ = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, img_size, alpha=0.0)
        intrinsics[:3, :3] = ncm
        map_x, map_y = cv2.initUndistortRectifyMap(intrinsics, distortion, None, ncm, img_size, cv2.CV_32FC1)
        img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    T_cam2pix = np.insert(np.insert(intrinsics, 3, [0, 0, 0], axis=1), 3, [0, 0, 0, 1], axis=0)
    T_ldr2cam_4x4 = np.insert(T_ldr2cam, 3, [0, 0, 0, 1], axis=0)
    T_ldr2pix = T_cam2pix @ T_ldr2cam_4x4

    # 8 corners of a box in its own local frame, then rotated+translated into
    # the ego/lidar frame -- same layout as dataset.draw_3d_box_in_cylinder.
    edges = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    def corners_3d(x, y, z, l, w, h, theta):
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                              [np.sin(theta), np.cos(theta), 0],
                              [0, 0, 1]])
        local = np.array([[l/2, w/2, h/2], [l/2, w/2, -h/2], [l/2, -w/2, h/2], [l/2, -w/2, -h/2],
                          [-l/2, w/2, h/2], [-l/2, w/2, -h/2], [-l/2, -w/2, h/2], [-l/2, -w/2, -h/2]])
        return local @ rotation.T + np.array([x, y, z])

    def draw_box(corners, color, thickness=2):
        pts_h = np.insert(corners, 3, 1, axis=1).T   # homogeneous coords, (4, 8)
        proj = T_ldr2pix @ pts_h                       # (4, 8)
        depths = proj[2, :]
        if np.any(depths <= 0.1):
            return False  # box is behind/straddling the camera plane -- skip it
        px = (proj[0, :] / depths).astype(int)
        py = (proj[1, :] / depths).astype(int)
        for a, b in edges:
            cv2.line(img, (px[a], py[a]), (px[b], py[b]), color, thickness)
        return True

    n_gt_drawn, n_pred_drawn = 0, 0
    for cls_name, _logit, (x, y, z, th, l, w, h), _trk in dict_item['label'][0]:
        if draw_box(corners_3d(x, y, z, l, w, h, th), color=(0, 255, 0)):  # BGR green
            n_gt_drawn += 1

    pred_dicts = dict_item['pred_dicts'][0]
    pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
    pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
    for i in range(len(pred_scores)):
        if pred_scores[i] <= conf_thr:
            continue
        x, y, z, l, w, h, th = pred_boxes[i]
        if draw_box(corners_3d(x, y, z, l, w, h, th), color=(0, 0, 255)):  # BGR red
            n_pred_drawn += 1

    img = _draw_legend_on_image(img, [
        ("GT box", (0, 255, 0)),
        (f"Prediction (score>{conf_thr})", (0, 0, 255)),
    ])

    summary = f"{n_gt_drawn} GT boxes, {n_pred_drawn} pred boxes drawn on image"
    if out_path is not None:
        cv2.imwrite(out_path, img)
        print(f"saved -> {out_path}  ({summary})")
    else:
        print(f"camera panel: {summary}")

    if return_array:
        return img


def show_open3d_interactive(dict_item, dataset, conf_thr: float = 0.5):
    """
    A real, visible, interactive open3d window: rotate with left-drag, pan
    with right-drag, zoom with the scroll wheel -- exactly like
    main_test_0.py's vis.run(). This BLOCKS until you close the window
    (or press 'q'), then the script continues.

    Colors: GT boxes are green, predictions are red -- same convention as
    the BEV and camera views. (The original repo colors boxes by class
    instead, which makes GT and predictions of the same class look
    identical; we use color to mean GT-vs-prediction here instead, since
    that's the more useful distinction when reading a detection result.)

    Requires an actual display session (X11/Wayland) -- won't work over
    plain SSH with no X forwarding.
    """
    import open3d as o3d

    pc_lidar = dataset.get_ldr64_from_path(dict_item['meta'][0]['path']['ldr64'])

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_lidar[:, :3])
    vis.add_geometry(pcd)

    n_gt, n_pred = 0, 0
    for cls_name, _logit, (x, y, z, th, l, w, h), _trk in dict_item['label'][0]:
        dataset.draw_3d_box_in_cylinder(vis, (x, y, z), th, l, w, h, color=[0, 1, 0], radius=0.08)
        n_gt += 1

    pred_dicts = dict_item['pred_dicts'][0]
    pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
    pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
    for i in range(len(pred_scores)):
        if pred_scores[i] <= conf_thr:
            continue
        x, y, z, l, w, h, th = pred_boxes[i]
        dataset.draw_3d_box_in_cylinder(vis, (x, y, z), th, l, w, h, color=[1, 0, 0], radius=0.03)
        n_pred += 1

    print(f"Opening interactive open3d window -- {n_gt} GT boxes [green], "
          f"{n_pred} pred boxes [red]. Close the window (or press 'q') to continue.")
    vis.run()
    vis.destroy_window()


def save_combined_view(dict_item, dataset, out_path: str, cam_path: str,
                        calib_yml: str = None, conf_thr: float = 0.5):
    """One PNG: the BEV panel and the camera-projection panel, side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2

    fig, (ax_bev, ax_cam) = plt.subplots(1, 2, figsize=(14, 7))

    visualize_bev_detections(dict_item, dataset, conf_thr=conf_thr, ax=ax_bev)

    cam_img_bgr = project_boxes_to_image(dict_item, dataset, cam_path, out_path=None,
                                          calib_yml=calib_yml, conf_thr=conf_thr,
                                          return_array=True)
    ax_cam.imshow(cv2.cvtColor(cam_img_bgr, cv2.COLOR_BGR2RGB))
    ax_cam.set_title("Camera projection")
    ax_cam.axis('off')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {out_path}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", default="configs/cfg_RTNH_wide.yml")
    ap.add_argument("--ckpt", default="pretrained/RTNH_wide_10.pt")
    ap.add_argument("--idx", type=int, default=0, help="test-set sample index")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--conf-thr", type=float, default=0.5,
                     help="score threshold for drawing/counting a prediction as 'detected'")

    # Mode 1: feature extraction (stop before the detection head)
    ap.add_argument("--extract-bev-feat", action="store_true",
                     help="stop after the backbone and dump dict_item['bev_feat']")
    ap.add_argument("--visualize", action="store_true",
                     help="also save a heatmap PNG of bev_feat (needs --extract-bev-feat)")

    # Mode 2: full detection + visualization (mix and match any of these)
    ap.add_argument("--visualize-detections", action="store_true",
                     help="save a BEV point-cloud+box PNG (GT green, predictions red)")
    ap.add_argument("--project-cam", action="store_true",
                     help="project boxes onto a camera image (auto-resolved "
                          "unless --cam-img is given explicitly)")
    ap.add_argument("--cam-img", default=None,
                     help="explicit path to a camera image; overrides auto-resolution")
    ap.add_argument("--cam-folder", default="cam-front",
                     help="folder name to search when auto-resolving the camera image")
    ap.add_argument("--calib-yml", default=None,
                     help="per-sequence camera calibration yml (see project_boxes_to_image's "
                          "docstring); falls back to util_calib.dict_front0 if omitted")
    ap.add_argument("--open3d-interactive", action="store_true",
                     help="open a real, rotatable open3d window (blocks until closed). "
                          "Needs an actual display session, not plain SSH.")
    ap.add_argument("--combined", action="store_true",
                     help="save one PNG with the BEV and camera panels side by side")

    args = ap.parse_args()

    # The devkit's own package imports (models., datasets., utils.) are
    # relative to the repo root, so this script must run with the K-Radar
    # repo as cwd.
    sys.path.insert(0, str(Path.cwd()))

    cfg, dataset_test, network = build_from_scratch(args.cfg, args.ckpt, args.device)

    subset = Subset(dataset_test, [args.idx])
    loader = torch.utils.data.DataLoader(
        subset, batch_size=1, shuffle=False,
        collate_fn=dataset_test.collate_fn,
        num_workers=0,  # keep 0 while debugging -- real stack traces, no worker hop
    )
    dict_item = next(iter(loader))

    detection_mode_requested = (args.visualize_detections or args.cam_img
                                 or args.project_cam or args.open3d_interactive
                                 or args.combined)

    if args.extract_bev_feat:
        out = run_partial_forward(network, dict_item, stop_after="backbone")
        bev_feat = out["bev_feat"]
        print(f"bev_feat: shape={tuple(bev_feat.shape)}, dtype={bev_feat.dtype}, "
              f"device={bev_feat.device}")
        torch.save(bev_feat.detach().cpu(), f"bev_feat_idx{args.idx}.pt")
        print(f"saved -> bev_feat_idx{args.idx}.pt")
        if args.visualize:
            visualize_bev_feat(bev_feat, f"bev_feat_idx{args.idx}.png")

    elif detection_mode_requested:
        dict_item = run_full_forward(network, dict_item)
        print_detection_summary(dict_item, dataset_test, conf_thr=args.conf_thr)

        # Only bother resolving a camera path if something actually needs one.
        cam_path = None
        if args.cam_img or args.project_cam or args.combined:
            cam_path = args.cam_img or resolve_camera_image(dict_item, cam_folder=args.cam_folder)
            if args.cam_img is None:
                print(f"auto-resolved camera image -> {cam_path}")

        if args.visualize_detections:
            visualize_bev_detections(dict_item, dataset_test,
                                      out_path=f"detections_bev_idx{args.idx}.png",
                                      conf_thr=args.conf_thr)
        if args.cam_img or args.project_cam:
            project_boxes_to_image(dict_item, dataset_test, cam_path,
                                    out_path=f"detections_cam_idx{args.idx}.png",
                                    calib_yml=args.calib_yml, conf_thr=args.conf_thr)
        if args.combined:
            save_combined_view(dict_item, dataset_test,
                                out_path=f"detections_combined_idx{args.idx}.png",
                                cam_path=cam_path, calib_yml=args.calib_yml,
                                conf_thr=args.conf_thr)
        if args.open3d_interactive:
            # Runs last since vis.run() blocks until the window is closed --
            # everything above has already been printed/saved by this point.
            show_open3d_interactive(dict_item, dataset_test, conf_thr=args.conf_thr)

    else:
        out = run_full_forward(network, dict_item)
        print("output dict keys:", list(out.keys()))


if __name__ == "__main__":
    # spconv/open3d's native CUDA teardown can race with Python's own cleanup
    # and crash on exit with a harmless-but-alarming-looking "free(): invalid
    # pointer" / "Aborted". Forcing a clean process exit here (both on success
    # and on error) means that crash never gets the chance to happen.
    exit_code = 0
    try:
        main()
    except SystemExit as e:
        exit_code = e.code or 0
    except BaseException:
        import traceback
        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
