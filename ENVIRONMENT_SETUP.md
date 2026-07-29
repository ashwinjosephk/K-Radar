# K-Radar Modern Environment Setup (pip + Python 3.9, tested on Ubuntu, RTX 3080/3060, driver 535.x / CUDA 12.2)

The official `docs/detection.md` env (conda, Python 3.8.13, torch 1.11.0+cu113, spconv-cu113) is no
longer reproducible as-is: several pinned packages are no longer published, and parts of the code
have drifted ahead of the pinned versions. This doc captures a working alternative, pip-only, and
every code patch needed to get from a clean clone to a running `main_vis.py` GUI on modern hardware/drivers.

## 1. Why the original pins don't work anymore

| Original pin | Problem |
|---|---|
| Python 3.8.13 | `spconv-cu1xx` wheels only publish for Python ≥3.9 |
| `spconv-cu113` | Never existed on PyPI — `spconv` only ships `cu114`, `cu118`, `cu121`, `cu124` |
| `open3d==0.15.1` | Never had a pip wheel (conda-only release) |
| `open3d==0.15.2` | Only ships wheels for Python 3.8–3.9 (caps how high you can go) |
| `opencv-python==4.2.0.32` | No wheel for Python ≥3.9 (predates cp39 builds) |
| torch 1.11.0 | `torch._six` (used in `utils/util_optim.py`) was removed in torch 2.0 |
| torch 1.11 / torchvision 0.12 | `models/backbone_2d/image_backbone.py` uses `ResNet18_Weights` + Swin models, which need torchvision ≥0.13 (→ torch ≥1.12) |

**Net effect:** Python 3.9 is the only version that satisfies both `spconv` (≥3.9) and `open3d` (≤3.9).
And since the image backbone needs a modern torchvision anyway, the whole stack has to move forward
together rather than trying to preserve the original 1.11/0.15.1/cu113 trio.

## 2. Verified working versions

- Python 3.9.18 (via `pyenv`, plain `venv` — no conda needed)
- torch==2.1.2+cu121, torchvision==0.16.2+cu121 (pick `cuXXX` to match your driver's max CUDA via `nvidia-smi`; cu121 works for driver ≥530.x)
- spconv-cu121
- open3d==0.15.2 (closest pip-available match to the original 0.15.1)
- opencv-python==4.5.5.64 (closest pip-available match to the original 4.2.0.32; API is unchanged for what this repo uses)
- numpy<2 (numpy 2.0's ABI break breaks the older opencv-python wheel)
- setuptools<70 (needed for `torch.utils.cpp_extension`'s `pkg_resources` import when building custom CUDA ops)

## 3. Setup steps

```bash
# Python 3.9 via pyenv (no conda)
curl https://pyenv.run | bash
# add pyenv init to ~/.bashrc, restart shell
pyenv install 3.9.18
~/.pyenv/versions/3.9.18/bin/python -m venv kradar_env
source kradar_env/bin/activate

pip install --upgrade pip
pip install "setuptools<70"
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install spconv-cu121
pip install "numpy<2" open3d==0.15.2 easydict opencv-python==4.5.5.64 PyQt5 \
  scikit-image numba einops matplotlib scipy tqdm tensorboard nms SharedArray

# Build the two custom CUDA extensions (needs nvcc matching your torch's CUDA build, e.g. 12.1;
# set TORCH_CUDA_ARCH_LIST to your GPU's compute capability — 8.6 for RTX 30-series/Ampere)
cd utils/Rotated_IoU/cuda_op
TORCH_CUDA_ARCH_LIST="8.6" python setup.py install
cd ../../../ops
TORCH_CUDA_ARCH_LIST="8.6" python setup.py develop
cd ../..
```

## 4. Required code patches

All of these are genuine upstream bugs/version-drift, not environment mistakes — worth upstreaming
as PRs/issues, not just carrying as local patches.

### 4.1 `utils/util_optim.py` — `torch._six` removed in torch 2.0
```diff
- from torch._six import inf
+ inf = float('inf')
```

### 4.2 `models/head/rdr_spcube_head.py` — `nms` package API mismatch
The PyPI `nms` package (Tom Hoag's) never re-exports `rboxes` at the top-level `nms` namespace in any
released version (0.1.0–0.1.6) — it only lives in the `nms.nms` submodule. K-Radar's own
`docs/detection.md` step 6 documents patching the *installed package's* `__init__.py` instead; the
equivalent, cleaner fix is patching K-Radar's own import:
```diff
- import nms
+ import nms.nms as nms
```

### 4.3 The `nms` package itself needs one more patch (per `docs/detection.md` step 6)
In the installed `nms` package's `nms/nms.py`, inside `rboxes()`:
```diff
  for rrect in rrects:
+     rrect = tuple(rrect)
      r = cv2.boxPoints(rrect)
-     print(r)
+     # print(r)
      polys.append(r)
```
(`cv2.boxPoints` requires a tuple, not a list; the `print` is leftover debug spam.)

### 4.4 `uis/ui_vis.py` — `DIC_CLASS_BGR` typo
Line ~420, inside `pushButtonCameraVis`, uses a key that doesn't exist in any shipped config
(every config defines `CLASS_BGR`, and the *rest* of the file, e.g. line 370, already uses it correctly):
```diff
- color = self.cfg.VIS.DIC_CLASS_BGR[cls_name]
+ color = self.cfg.VIS.CLASS_BGR[cls_name]
```
(Same typo also appears — commented out — near lines 575/594/614, and a sibling bug
`DIC_CLASS_RGB` exists in `utils/util_dataset.py`'s `func_show_lidar_point_cloud`, not yet hit/fixed.)

### 4.5 `uis/ui_vis.py` — Qt platform-plugin collision (`cv2` vs `PyQt5`)
`opencv-python`'s Linux wheel unconditionally overwrites `QT_QPA_PLATFORM_PLUGIN_PATH` at import time
(see its `cv2/config-3.py`), pointing it at cv2's own bundled Qt plugins — which aren't ABI-compatible
with the PyQt5 `QApplication` already running in this GUI app. Fix: reset the variable back to
PyQt5's own plugins immediately after `import cv2`:
```diff
  import cv2
+ import PyQt5
+ os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(
+     os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins', 'platforms'
+ )
  import open3d as o3d
```

### 4.6 `cv2.imshow()` calls crash regardless of 4.5 — replace with `imwrite`
Even after 4.5, any `cv2.imshow()` call in this codebase still crashes the same way, because it's not
actually a path problem — it's two independently-compiled Qt runtimes (cv2's bundled one and PyQt5's)
both trying to own the platform integration in one process. There is no environment-level fix; the
practical fix is to not use `cv2.imshow` inside a PyQt5 app at all. Affected, confirmed call sites:

- `uis/ui_vis.py`, `pushButtonCameraVis` (~line 435):
```diff
- cv2.imshow('front_iamge', cv_img)
- cv2.waitKey(0)
- cv2.destroyAllWindows()
+ out_path = '/tmp/kradar_camera_vis.png'
+ cv2.imwrite(out_path, cv_img)
+ print(f"Saved camera visualization to {out_path}")
```

- `utils/util_dataset.py`, `func_show_radar_tensor_bev` (~lines 118-129):
```diff
  if not (bboxes is None):
      arr_yx_bbox = arr_yx_bbox.transpose((1,0,2))
      arr_yx_bbox = np.flip(arr_yx_bbox, axis=(0,1))
-     cv2.imshow('Cartesian (bbox)', cv2.resize(arr_yx_bbox,(0,0),fx=4,fy=4))
+     out_img = cv2.resize(arr_yx_bbox,(0,0),fx=4,fy=4)
  else:
      arr_yx = arr_yx.transpose((1,0,2))
      arr_yx = np.flip(arr_yx_bbox, axis=(0,1))
-     cv2.imshow('Cartesian (bbox)', cv2.resize(arr_yx,(0,0),fx=2,fy=2))
+     out_img = cv2.resize(arr_yx,(0,0),fx=2,fy=2)
-
- cv2.waitKey(0)
+ cv2.imwrite('/tmp/kradar_radar_bev_cartesian.png', out_img)
+ print("Saved radar BEV (Cartesian) to /tmp/kradar_radar_bev_cartesian.png")
```

There are further un-patched `cv2.imshow` calls elsewhere in `utils/util_dataset.py` (CFAR
visualization, radar-cube slicing tools) — same fix pattern applies if/when those functions are used.

**Known gap / TODO:** this file-based workaround means every visualization button requires manually
opening a saved PNG afterward instead of popping up inline. A cleaner fix (not yet implemented) would
render results into the existing `QLabel` widgets the GUI already uses for the camera thumbnail
preview, which stays inside PyQt5's own Qt runtime and avoids the collision entirely — no `cv2.imshow`
or `cv2.imwrite`-and-reopen required.

## 5. Dataset layout note
If you extract sequence zips (e.g. `1.zip`) into a folder of the same name (e.g. `Scenes/1/`), you get
an extra nesting level (`Scenes/1/1/cam-front`, etc.), since each zip already contains a top-level
folder named after its sequence number. `DATASET.DIR.LIST_DIR` in the config must point to the
**parent whose immediate children are the sequence-number folders** — e.g. `Scenes/1` if your data
sits at `Scenes/1/1/...`, or just `Scenes` if you unzip future sequences directly into `Scenes/`
without the extra wrapper folder.
