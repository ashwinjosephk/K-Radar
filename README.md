# K-Radar (fork — modern pip/Python 3.9 environment)

This fork exists because the original repo's environment (conda, Python 3.8.13, torch
1.11.0+cu113) is no longer reproducible as-is: several pinned dependency versions have since
been pulled from PyPI, and parts of the code have drifted ahead of the pinned versions (removed
torch APIs, torchvision APIs that didn't exist yet, a few outright typos). This fork documents a
working pip-only environment on modern hardware/drivers, and patches every bug that showed up
getting there.

**Start here:** [`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md) — full explanation of
what broke and why, the verified working dependency versions, and every code patch with a diff.
`requirements.txt` in this fork is already updated to match.

## What's confirmed working in this fork

- `main_vis.py` GUI — Sensor Visualization tab: **Camera Front Vis**, **Radar Tensor Vis**
  (range-azimuth, both polar and Cartesian BEV), **Lidar Point Cloud Vis** — all popping up as
  native windows (see `ENVIRONMENT_SETUP.md` §4.6 for the PyQt5/OpenCV Qt-collision fix that
  made this possible)
- `main_test_0.py` — RTNH_wide pretrained inference, GT (gray) vs. predicted (colored) boxes in
  an Open3D viewer
- `datasets/kradar_detection_v1_1.py` — sparse radar cube generation from raw `radar_zyx_cube`,
  at arbitrary quantile densities (see below)

Not yet attempted in this fork: odometry visualization, tracking, sensor fusion (ASF)
train/eval, auto-labeling. See `ENVIRONMENT_SETUP.md` and the original `docs/` for what's
realistically runnable vs. what points to external/unreleased repos.

## Dataset layout — where everything actually lives

This fork was set up and tested against **sequence 1 only** (not the full 58-sequence, ~15TB
release). Three separate things live in three separate places — worth keeping straight, since
they're easy to conflate:

```
<your data root>/
├── Scenes/
│   └── 1/                          ← from the official sequence zip (e.g. `1.zip`)
│       └── 1/
│           ├── cam-front/, cam-left/, cam-rear/, cam-right/
│           ├── info_calib/
│           ├── info_label/         ← object detection ground truth (per-frame .txt, GT boxes)
│           ├── os1-128/, os2-64/   ← raw LiDAR point clouds
│           ├── radar_tesseract/    ← raw 4D (Doppler/Range/Azimuth/Elevation) radar tensor
│           ├── radar_zyx_cube/     ← same data, converted to dense Cartesian — the INPUT for
│           │                         all sparse-radar generation below
│           ├── sparse_cube/, sparse_cube_30/
│           └── description.txt
│
├── sparse_radar_tensor_wide_range-001/    ← pre-made sparse radar data, downloaded separately
│   └── sparse_radar_tensor_wide_range/       from KAIST's Google Drive (docs/preprocessing.md)
│       └── rtnh_wider_1p_1/1/                — this is what main_test_0.py / cfg_RTNH_wide.yml
│           └── sprdr_00033.npy, ...            point `rdr_sparse.dir` at for pretrained inference
│
├── my_sparse_gen/                         ← self-generated sparse radar data, produced locally
│   ├── rtnh_wider_1p_1/1/                    by datasets/kradar_detection_v1_1.py, from this
│   │   └── sprdr_00033.npy, ...              fork's own radar_zyx_cube — NOT downloaded
│   ├── rtnh_wider_5p_1/1/                    (QUANTILE_RATE: 0.01 / 0.05 / 0.10 respectively —
│   │   └── sprdr_00033.npy, ...              see configs/sparse_rdr_data_generation/)
│   └── rtnh_wider_10p_1/1/
│       └── sprdr_00033.npy, ...
│
└── RTNH_wide_10.pt                        ← pretrained checkpoint, downloaded separately
                                               (Google Drive link in ENVIRONMENT_SETUP.md)
```

Ground truth, specifically — there are two unrelated kinds, don't confuse them:

- **Object detection GT** (car/pedestrian boxes) → `Scenes/1/1/info_label/*.txt`, used
  everywhere in this fork so far (Camera/Radar/Lidar Vis, RTNH's gray boxes)
- **Odometry GT** (ego-vehicle trajectory, KITTI-format 3×4 pose matrices from LeGO-LOAM) →
  ships inside the repo itself at `resources/odometry/gt/gt_01.txt`, **not** downloaded, **not**
  in `Scenes` — a completely separate thing, not yet used in this fork
- A third kind, the crowd-refined "v2.0"/"v2.1" relabeling, is deliberately **not** used in this
  fork (`cfg_RTNH_wide.yml`'s `label_version` was switched from `v2_0` to `v1_0` specifically to
  avoid needing it — see `ENVIRONMENT_SETUP.md`)

### Config → path mapping (which config points where)

| Config | Key(s) | Points at |
|---|---|---|
| `configs/cfg_GUI_TOOL.yml` | `DATASET.DIR.LIST_DIR` | `Scenes/1` |
| `configs/cfg_RTNH_wide.yml` | `DATASET.path_data.list_dir_kradar` | `Scenes/1` |
| `configs/cfg_RTNH_wide.yml` | `DATASET.rdr_sparse.dir` | `sparse_radar_tensor_wide_range-001/.../rtnh_wider_1p_1` |
| `configs/sparse_rdr_data_generation/cfg_gen_wider_rtnh_*p.yml` | `DATASET.DIR.LIST_DIR` | `Scenes/1` |
| `configs/sparse_rdr_data_generation/cfg_gen_wider_rtnh_*p.yml` | `SPARSE_DATA.SAVE_FOLDER` | `my_sparse_gen` |
| `main_test_0.py` | `PATH_MODEL` | `RTNH_wide_10.pt` |

## Original K-Radar documentation

Everything else — dataset background, benchmark details, the full README with paper links and
demo videos — is unchanged from upstream. See [`readme.md`](readme.md) and [`docs/`](docs/).
