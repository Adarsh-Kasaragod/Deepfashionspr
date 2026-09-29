import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def visualize_sample(image_path, anno_path, output_path):
    # Load image
    img = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(img)

    # Load annotations
    with open(anno_path, 'r') as f:
        anno = json.load(f)

    # Draw each item in the annotation
    for key, item in anno.items():
        if key in ['source', 'pair_id']:
            continue
        
        # Bounding box [x1, y1, x2, y2]
        bbox = item['bounding_box']
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Draw bounding box
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Category label
        cat_name = item['category_name']
        plt.text(x1, y1 - 5, cat_name, color='red', fontsize=12, backgroundcolor='white')
        
        # Landmarks [x1, y1, v1, x2, y2, v2, ...]
        landmarks = item['landmarks']
        points_x = landmarks[0::3]
        points_y = landmarks[1::3]
        points_v = landmarks[2::3]
        
        for px, py, pv in zip(points_x, points_y, points_v):
            if pv > 0: # If visible or occluded (but labeled)
                ax.plot(px, py, 'go', markersize=4) # green dots for landmarks

    plt.axis('off')
    plt.title('DeepFashion2 Ground Truth Visualization')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    DATA_ROOT = r"D:\DeepFashion2-master\DeepFashion2-master"
    # Take the first validation image
    img_path = os.path.join(DATA_ROOT, "validation", "image", "000001.jpg")
    ann_path = os.path.join(DATA_ROOT, "validation", "annos", "000001.json")
    out_path = r"D:\DeepFashion2-master\sample_output.jpg"
    
    if os.path.exists(img_path) and os.path.exists(ann_path):
        visualize_sample(img_path, ann_path, out_path)
    else:
        print("Data files not found. Please ensure the paths are correct.")
