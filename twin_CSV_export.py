from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np
from ultralytics import YOLO
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pillow_heif import register_heif_opener
from concurrent.futures import ThreadPoolExecutor

register_heif_opener()
load_dotenv()

# Config
HISTORICAL_DATA_DIR = "./historical_data"
RESULTS_DIR = "./historical_data_results"
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.20'))
IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', '0.20'))

# Multic core usage
NUM_CORES = os.cpu_count()  

# YOLO batch size, set on your preference / device 
YOLO_BATCH_SIZE = int(os.getenv('YOLO_BATCH_SIZE', '16'))

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load YOLO 
yolo_model = YOLO('./runs/detect/train9/weights/last.pt')
print(f"Using {NUM_CORES} CPU cores | YOLO batch size: {YOLO_BATCH_SIZE}")


def extract_datetime_from_image(image_path):
    """Extract datetime from EXIF. Returns None if not found (image will be skipped)."""
    try:
        image = Image.open(image_path)
        exif_data = None

        if hasattr(image, '_getexif'):
            try:
                exif_data = image._getexif()
            except Exception:
                pass

        if exif_data is None:
            try:
                exif_info = image.getexif()
                if exif_info:
                    exif_data = dict(exif_info)
                    exif_ifd = exif_info.get_ifd(0x8769)
                    if exif_ifd:
                        exif_data.update(exif_ifd)
            except Exception:
                pass

        if exif_data:
            for tag_id, value in exif_data.items():
                if TAGS.get(tag_id) == "DateTimeOriginal":
                    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
            for tag_id, value in exif_data.items():
                if TAGS.get(tag_id) == "DateTime":
                    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")

    except Exception as e:
        print(f"  EXIF error for {os.path.basename(image_path)}: {e}")

    return None


def load_image(image_path):
    """Load and normalise image to RGB. Returns None on failure."""
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"  Failed to load {os.path.basename(image_path)}: {e}")
        return None


# ============== MAIN ==============
# heic = bad dont use it :(

supported_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.tif', '.heic')
image_files = sorted([
    f for f in os.listdir(HISTORICAL_DATA_DIR)
    if f.lower().endswith(supported_extensions) and not f.startswith('.')
])

if not image_files:
    print(f"No images found in {HISTORICAL_DATA_DIR}")
    exit()

print(f"Found {len(image_files)} images\n")

# Extract all EXIF dates in parallel 
print(f"Extracting EXIF timestamps in parallel ({NUM_CORES} threads)...")
image_paths = [os.path.join(HISTORICAL_DATA_DIR, f) for f in image_files]

with ThreadPoolExecutor(max_workers=NUM_CORES) as pool:
    timestamps = list(pool.map(extract_datetime_from_image, image_paths))

# Filter out images with no timestamp up front
valid = [
    (path, fname, ts)
    for path, fname, ts in zip(image_paths, image_files, timestamps)
    if ts is not None
]
skipped = len(image_files) - len(valid)
print(f"  {len(valid)} valid | {skipped} skipped (no timestamp)\n")

# Load all valid images in parallel 
print(f"Loading images in parallel ({NUM_CORES} threads)...")
valid_paths  = [v[0] for v in valid]
valid_fnames = [v[1] for v in valid]
valid_ts     = [v[2] for v in valid]

with ThreadPoolExecutor(max_workers=NUM_CORES) as pool:
    loaded_images = list(pool.map(load_image, valid_paths))

# Drop any that failed to load
final = [
    (img, fname, ts)
    for img, fname, ts in zip(loaded_images, valid_fnames, valid_ts)
    if img is not None
]
print(f"  {len(final)} images ready for inference\n")

# Batched YOLO inference on Metal GPU
print(f"Running YOLO on MPS (batch_size={YOLO_BATCH_SIZE})...")

all_images = [item[0] for item in final]
all_counts = []

for batch_start in range(0, len(all_images), YOLO_BATCH_SIZE):
    batch = all_images[batch_start:batch_start + YOLO_BATCH_SIZE]
    batch_results = yolo_model.predict(
        batch,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        device='mps',   # Metal GPU on Apple Silicon
        verbose=False,
    )
    all_counts.extend(len(r.boxes) for r in batch_results)

    end = min(batch_start + YOLO_BATCH_SIZE, len(all_images))
    print(f"  Processed {end}/{len(all_images)}")

# Assemble results
results_data = []
for (_, fname, ts), count in zip(final, all_counts):
    results_data.append({
        'timestamp':     ts.strftime('%Y-%m-%d %H:%M:%S'),
        'total_insects': count,
        'suzukii':       None,  # placeholder for future model
    })
    print(f"  {fname} | {ts.strftime('%Y-%m-%d %H:%M')} | insects: {count}")

# Save CSV
df = pd.DataFrame(results_data).sort_values('timestamp').reset_index(drop=True)
csv_path = os.path.join(RESULTS_DIR, 'insect_data.csv')
df.to_csv(csv_path, index=False)

print(f"\nDone. {len(df)} images processed, {skipped} skipped.")
print(f"CSV saved to: {csv_path}")