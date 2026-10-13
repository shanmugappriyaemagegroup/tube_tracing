
import cv2, numpy as np, glob
found = False
for t in sorted(glob.glob('dataset_tiles/images/train/*.jpg')):
    lab = t.replace('images','labels').rsplit('.',1)[0]+'.txt'
    lines = open(lab).read().splitlines()
    if any(ln.startswith('2 ') for ln in lines):   # class 2 = tubing
        im = cv2.imread(t); h,w = im.shape[:2]
        for ln in lines:
            v = ln.split()
            p = (np.array(v[1:],float).reshape(-1,2)*[w,h]).astype(int)
            col = (0,0,255) if v[0]=='2' else (0,255,0)
            cv2.polylines(im,[p],True,col,2)
        cv2.imwrite('tile_check_tubing.jpg', im)
        print('saved from', t)
        found = True
        break
if not found:
    print('no tile with a tubing label found in train')
