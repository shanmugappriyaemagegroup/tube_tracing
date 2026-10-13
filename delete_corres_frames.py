import os

# Your folders
IMAGE_FOLDER = r"C:\Users\EmageVision\Documents\dataset\images"
LABEL_FOLDER = r"C:\Users\EmageVision\Documents\dataset\labels"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Get image filenames without extensions
image_names = set()

for filename in os.listdir(IMAGE_FOLDER):
    name, ext = os.path.splitext(filename)

    if ext.lower() in IMAGE_EXTENSIONS:
        image_names.add(name.lower())


print(f"Images found: {len(image_names)}")
print("\nChecking labels...\n")

kept = 0
deleted = 0

# Check labels
for filename in os.listdir(LABEL_FOLDER):

    if not filename.lower().endswith(".txt"):
        continue

    label_name = os.path.splitext(filename)[0]

    if label_name.lower() in image_names:
        print(f"KEEP   : {filename}")
        kept += 1
    else:
        print(f"DELETE : {filename}")

        label_path = os.path.join(LABEL_FOLDER, filename)
        os.remove(label_path)

        deleted += 1


print("\n========== DONE ==========")
print(f"Images found  : {len(image_names)}")
print(f"Labels kept   : {kept}")
print(f"Labels deleted: {deleted}")