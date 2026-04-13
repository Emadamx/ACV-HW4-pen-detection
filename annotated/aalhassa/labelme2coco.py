import json
import os
import base64
import io
import numpy as np
from pathlib import Path
from pycocotools import mask as mask_utils
from skimage import measure
from PIL import Image


def labelme_to_coco(json_dir, output_path):
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_map = {}
    image_id = 1
    annotation_id = 1

    json_files = sorted(Path(json_dir).glob("*.json"))

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        img_filename = data["imagePath"].lstrip("../")
        img_w = data["imageWidth"]
        img_h = data["imageHeight"]

        coco["images"].append({
            "id": image_id,
            "file_name": img_filename,
            "width": img_w,
            "height": img_h
        })

        for shape in data["shapes"]:
            label = shape["label"]
            shape_type = shape["shape_type"]
            points = shape["points"]

            # Register category if new
            if label not in category_map:
                category_map[label] = len(category_map) + 1
                coco["categories"].append({
                    "id": category_map[label],
                    "name": label,
                    "supercategory": "none"
                })

            segmentation = []
            x, y, w, h = 0, 0, 0, 0

            if shape_type == "polygon":
                flat = [coord for point in points for coord in point]
                segmentation = [flat]

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x, y = min(xs), min(ys)
                w, h = max(xs) - x, max(ys) - y

            elif shape_type == "mask":
                mask_data = shape.get("mask")
                if not mask_data:
                    print(f"  Warning: empty mask field in {json_file.name}, skipping shape")
                    continue

                try:
                    mask_bytes = base64.b64decode(mask_data)
                    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
                    mask_np = np.array(mask_img) > 0

                    if not mask_np.any():
                        # Mask is empty — fall back to points as a rectangle
                        if len(points) == 2:
                            (x1, y1), (x2, y2) = points[0], points[1]
                            x, y = min(x1, x2), min(y1, y2)
                            w, h = abs(x2 - x1), abs(y2 - y1)
                            segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
                            print(f"  Warning: empty mask in {json_file.name}, used points bbox fallback")
                        else:
                            print(f"  Warning: empty mask and no valid points in {json_file.name}, skipping")
                            continue
                    else:
                        contours = measure.find_contours(mask_np.astype(float), 0)
                        valid = [c for c in contours if len(c) * 2 >= 6]

                        if not valid:
                            # Contours failed — fall back to mask bounding box
                            rows = np.any(mask_np, axis=1)
                            cols = np.any(mask_np, axis=0)
                            y1, y2 = np.where(rows)[0][[0, -1]]
                            x1, x2 = np.where(cols)[0][[0, -1]]
                            segmentation = [[float(x1), float(y1), float(x2), float(y1),
                                            float(x2), float(y2), float(x1), float(y2)]]
                            print(f"  Warning: used mask bbox fallback for {json_file.name}")
                        else:
                            segmentation = []
                            for contour in valid:
                                flat = [coord for point in contour for coord in [point[1], point[0]]]
                                segmentation.append(flat)

                        # Bbox from mask
                        rows = np.any(mask_np, axis=1)
                        cols = np.any(mask_np, axis=0)
                        y, y2 = np.where(rows)[0][[0, -1]]
                        x, x2 = np.where(cols)[0][[0, -1]]
                        w, h = float(x2 - x), float(y2 - y)
                        x, y = float(x), float(y)

                except Exception as e:
                    print(f"  Error processing mask in {json_file.name}: {e}")
                    continue

            elif shape_type == "rectangle":
                (x1, y1), (x2, y2) = points[0], points[1]
                x, y = min(x1, x2), min(y1, y2)
                w, h = abs(x2 - x1), abs(y2 - y1)
                segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]

            else:
                print(f"  Skipping unsupported shape type: {shape_type} in {json_file.name}")
                continue

            area = float(w * h) if not segmentation else compute_area(segmentation, img_h, img_w)

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_map[label],
                "bbox": [x, y, w, h],
                "area": area,
                "segmentation": segmentation,
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"\nDone! {len(coco['images'])} images, {len(coco['annotations'])} annotations, {len(coco['categories'])} categories")
    print("Categories:", {v["name"]: v["id"] for v in coco["categories"]})


def compute_area(segmentation, img_h, img_w):
    """Compute area from polygon segmentation using pycocotools."""
    try:
        rles = mask_utils.frPyObjects(segmentation, img_h, img_w)
        rle = mask_utils.merge(rles)
        return float(mask_utils.area(rle))
    except Exception:
        return 0.0


# Usage
labelme_to_coco(
    json_dir="./",
    output_path="annotations.json"
)