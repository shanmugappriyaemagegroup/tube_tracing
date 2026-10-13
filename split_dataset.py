
import os, random, shutil

random.seed(42)
src_img = r"C:\Users\EmageVision\Documents\training_dataset_head\images"
src_lbl =r"C:\Users\EmageVision\Documents\training_dataset_head\labels"
out = r"C:\Users\EmageVision\Documents\training_dataset_head\dataset"

for split in ['train', 'val']:
    os.makedirs(f'{out}/images/{split}', exist_ok=True)
    os.makedirs(f'{out}/labels/{split}', exist_ok=True)

images = [f for f in os.listdir(src_img) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(images)
split_idx = int(len(images) * 0.8)
train_imgs, val_imgs = images[:split_idx], images[split_idx:]

for split, imgs in [('train', train_imgs), ('val', val_imgs)]:
    for img in imgs:
        base = os.path.splitext(img)[0]
        shutil.copy(f'{src_img}/{img}', f'{out}/images/{split}/{img}')
        lbl = f'{base}.txt'
        if os.path.exists(f'{src_lbl}/{lbl}'):
            shutil.copy(f'{src_lbl}/{lbl}', f'{out}/labels/{split}/{lbl}')

print(f'Train: {len(train_imgs)}, Val: {len(val_imgs)}')
