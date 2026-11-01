"""
Depth video viewer + Z picker (needs only ffmpeg, no ffprobe).

Usage:
    python depth_viewer.py                      (uses VIDEO_PATH below)
    python depth_viewer.py my_depth_video.mkv

Controls:
    Left-click : print Z (metres) at that pixel in the terminal
    Space      : pause / resume
    any key    : start playback (first frame waits)
    q          : quit
"""
import re
import subprocess
import sys

import cv2
import numpy as np

VIDEO_PATH = r"C:\Users\EmageVision\Downloads\20261027_145529_648_depth.mkv"
DEPTH_SCALE = 0.001       # raw units -> metres (1 unit = 1 mm for most depth cameras)
NEAR, FAR = 200, 3000     # colour range for display only (raw units / mm)

path = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH

# ---- 1. Read video info with ffmpeg ---------------------------------------
probe = subprocess.run(["ffmpeg", "-hide_banner", "-i", path],
                       capture_output=True, text=True).stderr
video_lines = [l.strip() for l in probe.splitlines() if "Video:" in l]
if not video_lines:
    print(probe)
    sys.exit("No video stream found - check the file path.")

stream = video_lines[0]
print("Stream info:", stream)
if len(video_lines) > 1:
    print(f"Note: file has {len(video_lines)} video streams, using the first one:")
    for l in video_lines:
        print("   ", l)

W, H = map(int, re.search(r"\b(\d{2,5})x(\d{2,5})\b", stream).groups())
print(f"Resolution: {W}x{H}")

if not any(fmt in stream for fmt in ("gray16", "y16", "b16g", "gray12", "gray10")):
    print("\n*** WARNING: this stream does not look like 16-bit depth. "
          "If it is h264/hevc/yuv420p, the depth was stored as lossy 8-bit "
          "and real distances can NOT be recovered. ***\n")

# ---- 2. Stream raw 16-bit frames out of ffmpeg ----------------------------
proc = subprocess.Popen(
    ["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0",
     "-f", "rawvideo", "-pix_fmt", "gray16le", "-"],
    stdout=subprocess.PIPE)

FRAME_BYTES = W * H * 2
depth = None


def on_click(event, u, v, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and depth is not None:
        # median of a 5x5 patch, ignoring 0 (= no measurement)
        patch = depth[max(v - 2, 0):v + 3, max(u - 2, 0):u + 3].astype(np.float32)
        patch = patch[patch > 0]
        if patch.size:
            raw = np.median(patch)
            print(f"pixel ({u},{v}) -> raw {raw:.0f} -> Z = {raw * DEPTH_SCALE:.3f} m")
        else:
            print(f"pixel ({u},{v}) -> no depth (0)")


cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("depth", on_click)

frame_idx = 0
paused = False
while True:
    buf = proc.stdout.read(FRAME_BYTES)
    if len(buf) < FRAME_BYTES:
        print("End of video.")
        break
    depth = np.frombuffer(buf, np.uint16).reshape(H, W)

    if frame_idx == 0:
        valid = depth[depth > 0]
        if valid.size:
            print(f"Frame 0 raw values -> min {valid.min()}, max {valid.max()}, "
                  f"median {np.median(valid):.0f}, valid pixels {valid.size / depth.size:.0%}")
        else:
            print("Frame 0 has no non-zero pixels.")
        print("Click on the image to read Z. Press any key to play.")

    # visualisation only (does not change the depth values)
    vis = np.clip((depth.astype(np.float32) - NEAR) / (FAR - NEAR), 0, 1)
    vis = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis[depth == 0] = 0
    cv2.putText(vis, f"frame {frame_idx}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("depth", vis)

    wait = 0 if (frame_idx == 0 or paused) else 30
    key = cv2.waitKey(wait) & 0xFF
    if key == ord("q"):
        break
    if key == ord(" "):
        paused = not paused
        while paused:                       # stay on this frame, clicks still work
            k = cv2.waitKey(50) & 0xFF
            if k == ord(" "):
                paused = False
            elif k == ord("q"):
                proc.terminate()
                cv2.destroyAllWindows()
                sys.exit()
    frame_idx += 1

proc.terminate()
cv2.destroyAllWindows()