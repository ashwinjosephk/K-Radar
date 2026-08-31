"""
K-Radar template Q&A generator
================================
Phase 4: builds (question, answer) pairs directly from K-Radar's own
ground-truth labels — the same idea as nuScenes-QA, but templated
against K-Radar's box format instead of GPT-4-written captions.

Pure CPU, no GPU, no SECOND forward pass, no Qwen3 load needed — this
only touches the dataset class's label parsing, verified directly
against datasets/kradar_detection_v2_0.py's get_label():

    dict_item['meta']['label'] = list of
        (cls_name, (x, y, z, th, l, w, h), trk, avail)

Note: cfg_SECOND.yml has consider_cls=True, so `label` only ever
contains K-Radar's two "cared" classes — 'Sedan' and 'Bus or Truck'.
Pedestrians/bicycles/motorcycles exist in the raw files but are
filtered out before reaching us, for this specific config.
"""

# Same transformers/torch<2.2 pytree shim as llm_exp.py and
# main_test_0.py — must run before ANY import that could pull in
# torchvision, since `datasets/__init__.py` imports kradar_fusion_v1_0.py
# which imports torchvision directly, independently of the models/
# import chain that triggers the same crash elsewhere.
import torch.utils._pytree as _pytree
if not hasattr(_pytree, 'register_pytree_node'):
    def _register_pytree_node_shim(cls, flatten_fn, unflatten_fn, **kwargs):
        kwargs.pop('serialized_type_name', None)
        return _pytree._register_pytree_node(cls, flatten_fn, unflatten_fn)
    _pytree.register_pytree_node = _register_pytree_node_shim

import json
from utils.util_config import cfg, cfg_from_yaml_file
from datasets.kradar_detection_v2_0 import KRadarDetection_v2_0
from sample_config import SAMPLE_INDICES

PATH_CONFIG = './configs/cfg_SECOND.yml'
FORWARD_RANGE_M = 20.0  # "ahead" window for the existence question
FORWARD_HALF_WIDTH_M = 8.0
NEARBY_RADIUS_M = 30.0  # radius for the counting question


def obj_xy(obj):
    # obj = (cls_name, (x, y, z, th, l, w, h), trk, avail)
    return obj[1][0], obj[1][1]


def obj_dist(obj):
    x, y = obj_xy(obj)
    return (x ** 2 + y ** 2) ** 0.5


def make_qa_for_sample(sample):
    label = sample['meta']['label']

    qa_pairs = []

    # --- Existence ---
    ahead = [obj for obj in label
             if 0 <= obj_xy(obj)[0] <= FORWARD_RANGE_M
             and abs(obj_xy(obj)[1]) <= FORWARD_HALF_WIDTH_M]
    qa_pairs.append({
        "question": f"Is there a vehicle within {FORWARD_RANGE_M:.0f} meters ahead of the ego vehicle?",
        "answer": "Yes." if ahead else "No.",
    })

    # --- Counting ---
    nearby = [obj for obj in label if obj_dist(obj) <= NEARBY_RADIUS_M]
    qa_pairs.append({
        "question": f"How many vehicles are within {NEARBY_RADIUS_M:.0f} meters of the ego vehicle?",
        "answer": f"{len(nearby)}.",
    })

    # --- Nearest object class ---
    if label:
        nearest = min(label, key=obj_dist)
        nearest_dist = obj_dist(nearest)
        qa_pairs.append({
            "question": "What type of vehicle is closest to the ego vehicle?",
            "answer": f"{nearest[0]}.",
            "nearest_distance_m": round(nearest_dist, 1),   # for sanity-checking threshold calibration, not part of the answer text
        })
    else:
        qa_pairs.append({
            "question": "What type of vehicle is closest to the ego vehicle?",
            "answer": "There are no vehicles nearby.",
            "nearest_distance_m": None,
        })

    # --- Spatial: side of nearest object ---
    # These two are the point of the whole spatial experiment: unlike
    # existence/counting/class, they CANNOT be answered from an unordered
    # bag of BEV features — you have to know WHERE the activation is.
    # If shuffling the BEV grid doesn't hurt accuracy on these, the model
    # genuinely isn't using spatial structure.
    # K-Radar convention: +x = forward, +y = left (right-handed, z up).
    if label:
        nearest = min(label, key=obj_dist)
        nx, ny = obj_xy(nearest)
        qa_pairs.append({
            "question": "Is the closest vehicle to the left or the right of the ego vehicle?",
            "answer": "Left." if ny > 0 else "Right.",
            "nearest_y_m": round(ny, 1),
        })
        qa_pairs.append({
            "question": "Is the closest vehicle ahead of or behind the ego vehicle?",
            "answer": "Ahead." if nx > 0 else "Behind.",
            "nearest_x_m": round(nx, 1),
        })
    else:
        qa_pairs.append({
            "question": "Is the closest vehicle to the left or the right of the ego vehicle?",
            "answer": "There are no vehicles nearby.",
            "nearest_y_m": None,
        })
        qa_pairs.append({
            "question": "Is the closest vehicle ahead of or behind the ego vehicle?",
            "answer": "There are no vehicles nearby.",
            "nearest_x_m": None,
        })

    return qa_pairs


def main():
    cfg_loaded = cfg_from_yaml_file(PATH_CONFIG, cfg)
    dataset = KRadarDetection_v2_0(cfg=cfg_loaded, split='test')
    print(f"Dataset has {len(dataset)} samples (test split)")

    all_samples = []
    for idx in SAMPLE_INDICES:
        sample = dataset[idx]
        qa_pairs = make_qa_for_sample(sample)
        all_samples.append({
            "idx": idx,
            "seq": sample['meta']['seq'],
            "file_indices": sample['meta']['idx'],
            "num_obj": sample['meta']['num_obj'],
            "qa_pairs": qa_pairs,
        })
        print(f"\n--- sample idx {idx} (seq {sample['meta']['seq']}, {sample['meta']['num_obj']} objects) ---")
        print(f"  file indices: {sample['meta']['idx']}")
        for qa in qa_pairs:
            print(f"  Q: {qa['question']}")
            print(f"  A: {qa['answer']}", end='')
            extras = [f"{k.replace('nearest_','').replace('_m','')}={v}m"
                      for k in ('nearest_distance_m', 'nearest_y_m', 'nearest_x_m')
                      for v in [qa.get(k)] if v is not None]
            print(f"   ({', '.join(extras)})" if extras else "")

    with open('kradar_qa.json', 'w') as f:
        json.dump(all_samples, f, indent=2)
    print(f"\nSaved {len(all_samples)} samples' worth of Q&A pairs to kradar_qa.json")
    print(f"Indices used: {SAMPLE_INDICES}")
    print("These match llm_exp.py's Stage 1 bev_features/sample_<idx>.pt filenames exactly.")


if __name__ == '__main__':
    main()
