"""Convert CVAT for images 1.1 XML (polygons) to YOLO segmentation .txt files.

Each line: <class_id> x1 y1 x2 y2 ... (coords normalised 0-1 by image width/height)

Usage:
    python cvat_to_yolo_seg.py annotations.xml labels_out
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Adjust to match your dataset's class ids
CLASS_MAP = {
    "head": 0,
    "tail": 1,
}


def convert(xml_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = ET.parse(xml_path).getroot()
    n_files = n_polys = 0
    skipped = set()

    for img in root.iter("image"):
        w, h = float(img.get("width")), float(img.get("height"))
        lines = []
        for poly in img.iter("polygon"):
            label = poly.get("label")
            if label not in CLASS_MAP:
                skipped.add(label)
                continue
            coords = []
            for pt in poly.get("points").split(";"):
                x, y = map(float, pt.split(","))
                x = min(max(x / w, 0.0), 1.0)
                y = min(max(y / h, 0.0), 1.0)
                coords += [f"{x:.6f}", f"{y:.6f}"]
            lines.append(f"{CLASS_MAP[label]} " + " ".join(coords))
            n_polys += 1

        # One .txt per image (empty file if no objects = background image)
        (out / (Path(img.get("name")).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        n_files += 1

    print(f"Wrote {n_files} label files, {n_polys} polygons -> {out}")
    if skipped:
        print("Skipped unknown labels:", skipped)


if __name__ == "__main__":
    convert(sys.argv[1] if len(sys.argv) > 1 else "annotations.xml",
            sys.argv[2] if len(sys.argv) > 2 else "labels")