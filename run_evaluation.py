"""
DeepFashion2 Evaluation Runner
==============================
Uses the repo's custom cocoeval (294 keypoints) but replaces the broken
Linux _mask.so with the pip-installed Windows pycocotools.
"""

import sys, os, json, importlib, types

# ── Step 1: grab the pip-installed mask module (Windows binary) ───────────────
import pycocotools._mask as _mask_win
import pycocotools.mask  as mask_win

# ── Step 2: load the repo's coco.py & cocoeval.py as standalone modules ───────
REPO_PY = r"D:\DeepFashion2-master\deepfashion2_api\PythonAPI\pycocotools"

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Inject the pip mask module under the repo package name so relative imports work
pkg = types.ModuleType("deepfashion2_pycocotools")
pkg.mask   = mask_win
pkg._mask  = _mask_win
pkg.__path__ = [REPO_PY]
pkg.__package__ = "deepfashion2_pycocotools"
sys.modules["deepfashion2_pycocotools"] = pkg
sys.modules["deepfashion2_pycocotools.mask"]  = mask_win
sys.modules["deepfashion2_pycocotools._mask"] = _mask_win

# Patch the repo files to use our fake package name
def patched_load(fname, modname):
    src = open(os.path.join(REPO_PY, fname)).read()
    src = src.replace("from . import mask as maskUtils",
                      "from deepfashion2_pycocotools import mask as maskUtils")
    src = src.replace("from .mask import",
                      "from deepfashion2_pycocotools.mask import")
    src = src.replace("np.round((0.95 - .5) / .05) + 1", "int(np.round((0.95 - .5) / .05)) + 1")
    src = src.replace("np.round((1.00 - .0) / .01) + 1", "int(np.round((1.00 - .0) / .01)) + 1")
    src = src.replace("np.float", "float")
    spec = importlib.util.spec_from_loader(modname,
               importlib.machinery.SourceFileLoader(modname, os.path.join(REPO_PY, fname)))
    mod = types.ModuleType(modname)
    mod.__package__ = "deepfashion2_pycocotools"
    exec(compile(src, fname, "exec"), mod.__dict__)
    sys.modules[modname] = mod
    return mod

df2_coco     = patched_load("coco.py",     "deepfashion2_pycocotools.coco")
df2_cocoeval = patched_load("cocoeval.py", "deepfashion2_pycocotools.cocoeval")

COCO     = df2_coco.COCO
COCOeval = df2_cocoeval.COCOeval

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT    = r"D:\DeepFashion2-master\DeepFashion2-master"
EVAL_DIR     = r"D:\DeepFashion2-master\evaluation\example"
VAL_JSON_DIR = os.path.join(DATA_ROOT, "json_for_validation")

GT_COCO_JSON   = os.path.join(VAL_JSON_DIR, "deepfashion2_val_coco.json")
GT_KP_VIS_JSON = os.path.join(VAL_JSON_DIR, "keypoints_val_vis.json")

RES_BBOX_JSON  = os.path.join(EVAL_DIR, "example_bbox_results.json")
RES_KEYS_JSON  = os.path.join(EVAL_DIR, "example_keys_results.json")
RES_SEGM_JSON  = os.path.join(EVAL_DIR, "example_segm_results.json")

# ── Evaluation helper ─────────────────────────────────────────────────────────
def run_eval(gt_json, res_json, iou_type, label):
    print(f"\n{'='*60}")
    print(f"  {label}  [iouType='{iou_type}']")
    print(f"{'='*60}")
    for path, tag in [(gt_json, "GT"), (res_json, "Results")]:
        if not os.path.exists(path):
            print(f"  [SKIP] {tag} not found: {path}")
            return

    cocoGt   = COCO(gt_json)
    cocoDt   = cocoGt.loadRes(res_json)
    imgIds   = sorted(cocoGt.getImgIds())

    cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

# ── Run all three tasks ───────────────────────────────────────────────────────
run_eval(GT_COCO_JSON,   RES_BBOX_JSON,  "bbox",      "Clothes Detection")
run_eval(GT_KP_VIS_JSON, RES_KEYS_JSON,  "keypoints", "Landmark Estimation (visible only)")
run_eval(GT_COCO_JSON,   RES_SEGM_JSON,  "segm",      "Clothes Segmentation")

print("\n✅ All evaluations complete!")
