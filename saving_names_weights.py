import torch
src = r"C:\Users\EmageVision\Videos\best_21-10-2026.pt"
ckpt = torch.load(src, map_location="cpu", weights_only=False)
names = {0: "HEAD", 1: "TAIL"}
for k in ("model", "ema"):
    if ckpt.get(k) is not None:
        ckpt[k].names = names
dst = src.replace(".pt", "_named.pt")
torch.save(ckpt, dst)
print("saved", dst)