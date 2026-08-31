"""
Converts DeepPCB annotations to YOLO format.

Context: the raw DeepPCB dataset (and some Roboflow-hosted mirrors of 
PKU-Market-PCB) don't ship pre-converted YOLO-format labels, or in one 
case had corrupted class-name mappings. This script fixes/converts 
annotations into the standard YOLO .txt format expected by ultralytics.

See docs/training_results/metrics.md for the resulting model's evaluation.
"""
# 1. Re-download the dataset from Kaggle
!kaggle datasets download -d liuxiaolong1/pcb-defect-detection-dataset

# 2. Extract ONLY the DeepPCB folder
!unzip -qo pcb-defect-detection-dataset.zip "DeepPCB/*" -d /content/pcb_data

# 3. Clean up the zip file to save space
!rm pcb-defect-detection-dataset.zip

# 4. Run the safe conversion script
import os
from PIL import Image

def convert_deeepcb_safe(split):
    img_dir = f'/content/pcb_data/DeepPCB/{split}/images'
    lbl_dir = f'/content/pcb_data/DeepPCB/{split}/labels'
    
    converted_files = 0
    total_annotations = 0
    
    if not os.path.exists(img_dir):
        print(f"Directory not found: {img_dir}")
        return

    for img_name in os.listdir(img_dir):
        if not img_name.endswith(('.jpg', '.png')):
            continue
            
        img_path = os.path.join(img_dir, img_name)
        txt_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        lbl_path = os.path.join(lbl_dir, txt_name)
        
        if not os.path.exists(lbl_path):
            continue
            
        with Image.open(img_path) as img:
            w, h = img.size
            
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        yolo_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) >= 5:
                try:
                    x1, y1, x2, y2 = map(float, parts[:4])
                    cls_id = int(float(parts[4]))
                    
                    yolo_cls = cls_id - 1
                    if not (0 <= yolo_cls <= 5):
                        if 0 <= cls_id <= 5:
                            yolo_cls = cls_id
                        else:
                            continue
                            
                    box_w = (x2 - x1) / w
                    box_h = (y2 - y1) / h
                    center_x = (x1 + x2) / (2.0 * w)
                    center_y = (y1 + y2) / (2.0 * h)
                    
                    center_x = max(0.0, min(1.0, center_x))
                    center_y = max(0.0, min(1.0, center_y))
                    box_w = max(0.0, min(1.0, box_w))
                    box_h = max(0.0, min(1.0, box_h))
                    
                    yolo_lines.append(f"{yolo_cls} {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}")
                    total_annotations += 1
                except ValueError:
                    continue
                    
        with open(lbl_path, 'w') as out_f:
            out_f.write("\n".join(yolo_lines))
        if yolo_lines:
            converted_files += 1
            
    print(f"[{split}] Converted {converted_files} files with a total of {total_annotations} bounding boxes.")

convert_deeepcb_safe('train')
convert_deeepcb_safe('test')