
import json
import os
import numpy as np
from pathlib import Path

json_dir = "./"

for json_file in sorted(Path(json_dir).glob("*.json")):
    with open(json_file) as f:
        data = json.load(f)
    
    print(f"\n{json_file.name} — {len(data['shapes'])} shape(s)")
    for i, shape in enumerate(data["shapes"]):
        shape_type = shape["shape_type"]
        label = shape["label"]
        print(f"  [{i}] type={shape_type}, label={label}", end="")

        if shape_type == "mask":
            mask_data = shape.get("mask")
            if not mask_data:
                print(" ❌ SKIPPED — mask field is empty/missing")
                continue
            
            # Try decoding
            try:
                import base64, io
                import numpy as np
                from PIL import Image
                from skimage import measure

                mask_bytes = base64.b64decode(mask_data)
                mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
                mask_np = np.array(mask_img) > 0
                contours = measure.find_contours(mask_np, 0.5)
                valid = [c for c in contours if len(c) * 2 >= 6]
                
                if not valid:
                    print(f" ❌ SKIPPED — no valid contours found (total contours: {len(contours)})")
                else:
                    print(f" ✅ OK — {len(valid)} contour(s)")
            except Exception as e:
                print(f" ❌ SKIPPED — decode error: {e}")
        else:
            print(" ✅ OK")