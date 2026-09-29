import json
import os
from PIL import Image
import numpy as np

# ─── PATHS ────────────────────────────────────────────────────────────────────
DATA_ROOT    = r"D:\DeepFashion2-master\DeepFashion2-master"
VAL_ANNOS    = os.path.join(DATA_ROOT, "validation", "annos")
VAL_IMAGES   = os.path.join(DATA_ROOT, "validation", "image")
OUTPUT_JSON  = os.path.join(DATA_ROOT, "json_for_validation", "deepfashion2_val_coco.json")

# ─── DATASET SKELETON ─────────────────────────────────────────────────────────
CATEGORY_NAMES = [
    "short_sleeved_shirt", "long_sleeved_shirt", "short_sleeved_outwear",
    "long_sleeved_outwear", "vest", "sling", "shorts", "trousers", "skirt",
    "short_sleeved_dress", "long_sleeved_dress", "vest_dress", "sling_dress"
]

# Landmark ranges per category (start, end) — 0-indexed
CAT_KP_RANGES = {
    1:  (0,   25),
    2:  (25,  58),
    3:  (58,  89),
    4:  (89,  128),
    5:  (128, 143),
    6:  (143, 158),
    7:  (158, 168),
    8:  (168, 182),
    9:  (182, 190),
    10: (190, 219),
    11: (219, 256),
    12: (256, 275),
    13: (275, 294),
}

dataset = {
    "info": {"description": "DeepFashion2 Validation Set (COCO format)"},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
}

for cid, name in enumerate(CATEGORY_NAMES, start=1):
    dataset["categories"].append({
        "id": cid,
        "name": name,
        "supercategory": "clothes",
        "keypoints": [str(i) for i in range(1, 295)],
        "skeleton": []
    })

# ─── CONVERSION ───────────────────────────────────────────────────────────────
anno_files = sorted(os.listdir(VAL_ANNOS))
num_images = len(anno_files)
print(f"Converting {num_images} validation images to COCO format...")

sub_index = 0
errors = 0

for idx, fname in enumerate(anno_files):
    if not fname.endswith(".json"):
        continue

    num = int(fname.replace(".json", ""))
    json_path  = os.path.join(VAL_ANNOS,   fname)
    image_path = os.path.join(VAL_IMAGES,  fname.replace(".json", ".jpg"))

    if not os.path.exists(image_path):
        errors += 1
        continue

    try:
        imag = Image.open(image_path)
        width, height = imag.size
    except Exception:
        errors += 1
        continue

    with open(json_path, "r") as f:
        temp = json.load(f)

    pair_id = temp.get("pair_id", 0)

    dataset["images"].append({
        "coco_url": "",
        "date_captured": "",
        "file_name": fname.replace(".json", ".jpg"),
        "flickr_url": "",
        "id": num,
        "license": 0,
        "width": width,
        "height": height
    })

    for key, item in temp.items():
        if key in ("source", "pair_id"):
            continue

        sub_index += 1
        box       = item["bounding_box"]
        w, h      = box[2] - box[0], box[3] - box[1]
        bbox      = [box[0], box[1], w, h]
        cat       = item["category_id"]
        style     = item["style"]
        seg       = item["segmentation"]
        landmarks = item["landmarks"]

        points_x = np.array(landmarks[0::3])
        points_y = np.array(landmarks[1::3])
        points_v = np.array(landmarks[2::3])
        points   = np.zeros(294 * 3)

        if cat in CAT_KP_RANGES:
            start, end = CAT_KP_RANGES[cat]
            for n in range(start, end):
                local = n - start
                points[3 * n]     = points_x[local]
                points[3 * n + 1] = points_y[local]
                points[3 * n + 2] = points_v[local]

        num_points = int(np.sum(points_v > 0))

        dataset["annotations"].append({
            "area":           w * h,
            "bbox":           bbox,
            "category_id":    cat,
            "id":             sub_index,
            "pair_id":        pair_id,
            "image_id":       num,
            "iscrowd":        0,
            "style":          style,
            "num_keypoints":  num_points,
            "keypoints":      points.tolist(),
            "segmentation":   seg,
        })

    if (idx + 1) % 1000 == 0:
        print(f"  Processed {idx+1}/{num_images} images, {sub_index} annotations so far...")

print(f"\nDone! Total images: {len(dataset['images'])}, annotations: {len(dataset['annotations'])}, errors: {errors}")
print(f"Saving to: {OUTPUT_JSON}")

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(dataset, f)

print("Saved successfully!")
