# """
# RGB + depth pair viewer for OAK recordings.
# Click on the image -> X, Y, Z (metres) at that pixel, plus 3D distance between
# the last two clicks. Zoom and pan supported.

# Needs only ffmpeg (no ffprobe), numpy and opencv-python (NOT the headless build).

# Usage:
#     python rgbd_viewer.py
#     python rgbd_viewer.py <rgb_video> <depth_video>

# Mouse:
#     Left-click        : measure X, Y, Z at that pixel (2 clicks -> 3D distance)
#     Mouse wheel       : zoom in / out at the cursor
#     Right-button drag : pan when zoomed in
# Keys:
#     + / -   : zoom in / out (centre of view)
#     r       : reset zoom
#     c       : clear measured points
#     m       : switch view  RGB -> depth -> overlay
#     Space   : pause / resume (clicking and zooming work while paused)
#     q       : quit
# """
# import re
# import subprocess
# import sys

# import cv2
# import numpy as np

# # ---------------------------------------------------------------- settings --
# RGB_PATH = r"C:\Users\EmageVision\Downloads\20261027_145823_862_video.mp4"
# DEPTH_PATH = r"C:\Users\EmageVision\Downloads\20261027_145823_862_depth.mkv"
# DEPTH_SCALE = 0.001       # OAK depth is uint16 millimetres -> metres
# NEAR, FAR = 200, 3000     # colour range for depth display only (mm)

# # OAK RGB (CAM_A) intrinsics from the camera calibration, valid at 1920x1080
# FX, FY = 1070.789795, 1070.526978
# CX, CY = 962.309387, 550.0
# CALIB_W, CALIB_H = 1920, 1080

# if len(sys.argv) == 3:
#     RGB_PATH, DEPTH_PATH = sys.argv[1], sys.argv[2]


# # ------------------------------------------------------------------ probing --
# def probe(path):
#     """Return (width, height, fps, stream_line, duration) using ffmpeg -i."""
#     out = subprocess.run(["ffmpeg", "-hide_banner", "-i", path],
#                          capture_output=True, text=True).stderr
#     lines = [l.strip() for l in out.splitlines() if "Video:" in l]
#     if not lines:
#         print(out)
#         sys.exit(f"No video stream found in {path}")
#     line = lines[0]
#     w, h = map(int, re.search(r"\b(\d{2,5})x(\d{2,5})\b", line).groups())
#     fps_m = re.search(r"([\d.]+) fps", line)
#     fps = float(fps_m.group(1)) if fps_m else None
#     dur_m = re.search(r"Duration: ([\d:.]+)", out)
#     return w, h, fps, line, (dur_m.group(1) if dur_m else "?")


# rw, rh, rfps, rline, rdur = probe(RGB_PATH)
# dw, dh, dfps, dline, ddur = probe(DEPTH_PATH)
# print("RGB   stream:", rline, "| duration", rdur)
# print("Depth stream:", dline, "| duration", ddur)

# if not any(f in dline for f in ("gray16", "y16", "b16g", "gray12", "gray10")):
#     print("\n*** WARNING: depth stream is not 16-bit grayscale. If it is "
#           "h264/hevc/yuv420p, the real distances are lost. ***\n")
# if (rw, rh) != (dw, dh):
#     print(f"\nNOTE: RGB is {rw}x{rh} but depth is {dw}x{dh}. Depth will be "
#           "resized to the RGB size; only correct if depth was aligned to RGB.\n")
# if rfps and dfps and abs(rfps - dfps) > 0.5:
#     print(f"\nNOTE: fps differ (RGB {rfps}, depth {dfps}); frames may drift.\n")

# # Intrinsics are for 1920x1080; rescale if the RGB video is a different size
# sx, sy = rw / CALIB_W, rh / CALIB_H
# fx, fy, cx, cy = FX * sx, FY * sy, CX * sx, CY * sy
# if (rw, rh) != (CALIB_W, CALIB_H):
#     print(f"NOTE: intrinsics rescaled from {CALIB_W}x{CALIB_H} to {rw}x{rh}.")
# print(f"Intrinsics used: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")


# # ----------------------------------------------------------------- readers --
# def open_pipe(path, pix_fmt):
#     return subprocess.Popen(
#         ["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0",
#          "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
#         stdout=subprocess.PIPE)


# rgb_proc = open_pipe(RGB_PATH, "bgr24")
# dep_proc = open_pipe(DEPTH_PATH, "gray16le")
# RGB_BYTES = rw * rh * 3
# DEP_BYTES = dw * dh * 2

# # ------------------------------------------------------------------- state --
# depth = None              # current depth frame (uint16, RGB size)
# frame_idx = 0
# MODES = ["rgb", "depth", "overlay"]
# mode = 0
# points = []               # measured points: dict(u, v, X, Y, Z, frame)

# zoom = 1.0                # 1 = full image
# x0, y0 = 0, 0             # top-left of the visible crop, in image pixels
# drag = None               # (mouse_x, mouse_y, x0, y0) at right-button press
# MAX_ZOOM = 20.0


# def crop_size():
#     return max(1, int(round(rw / zoom))), max(1, int(round(rh / zoom)))


# def clamp_view():
#     global x0, y0
#     cw, ch = crop_size()
#     x0 = int(round(min(max(x0, 0), rw - cw)))
#     y0 = int(round(min(max(y0, 0), rh - ch)))


# def canvas_to_img(x, y):
#     """Mouse position on the displayed canvas -> pixel in the full image."""
#     cw, ch = crop_size()
#     u = x0 + x * cw / rw
#     v = y0 + y * ch / rh
#     return int(min(max(u, 0), rw - 1)), int(min(max(v, 0), rh - 1))


# def img_to_canvas(u, v):
#     cw, ch = crop_size()
#     return int((u - x0 + 0.5) * rw / cw), int((v - y0 + 0.5) * rh / ch)


# def zoom_at(x, y, factor):
#     """Zoom keeping the image point under canvas (x, y) fixed."""
#     global zoom, x0, y0
#     cw, ch = crop_size()
#     pu = x0 + x * cw / rw
#     pv = y0 + y * ch / rh
#     zoom = float(min(max(zoom * factor, 1.0), MAX_ZOOM))
#     cw, ch = crop_size()
#     x0 = pu - x * cw / rw
#     y0 = pv - y * ch / rh
#     clamp_view()


# def measure(u, v):
#     patch = depth[max(v - 2, 0):v + 3, max(u - 2, 0):u + 3].astype(np.float32)
#     patch = patch[patch > 0]
#     if not patch.size:
#         print(f"frame {frame_idx}  pixel ({u},{v}) -> no depth (0)")
#         return
#     Z = float(np.median(patch)) * DEPTH_SCALE
#     X = (u - cx) * Z / fx
#     Y = (v - cy) * Z / fy
#     points.append(dict(u=u, v=v, X=X, Y=Y, Z=Z, frame=frame_idx))
#     del points[:-2]       # keep the last two for the distance measurement
#     print(f"frame {frame_idx}  pixel ({u},{v}) -> "
#           f"X={X:+.3f}  Y={Y:+.3f}  Z={Z:.3f} m")
#     if len(points) == 2:
#         a, b = points
#         dist = np.sqrt((a["X"] - b["X"])**2 + (a["Y"] - b["Y"])**2 + (a["Z"] - b["Z"])**2)
#         print(f"   3D distance between last two points: {dist * 1000:.1f} mm")


# def on_mouse(event, x, y, flags, param):
#     global drag, x0, y0
#     if depth is None:
#         return
#     if event == cv2.EVENT_LBUTTONDOWN:
#         measure(*canvas_to_img(x, y))
#     elif event == cv2.EVENT_MOUSEWHEEL:
#         zoom_at(x, y, 1.25 if cv2.getMouseWheelDelta(flags) > 0 else 0.8)
#     elif event == cv2.EVENT_RBUTTONDOWN:
#         drag = (x, y, x0, y0)
#     elif event == cv2.EVENT_MOUSEMOVE and drag is not None and (flags & cv2.EVENT_FLAG_RBUTTON):
#         cw, ch = crop_size()
#         x0 = drag[2] - (x - drag[0]) * cw / rw
#         y0 = drag[3] - (y - drag[1]) * ch / rh
#         clamp_view()
#     elif event == cv2.EVENT_RBUTTONUP:
#         drag = None


# def colorize(d):
#     vis = np.clip((d.astype(np.float32) - NEAR) / (FAR - NEAR), 0, 1)
#     vis = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
#     vis[d == 0] = 0
#     return vis


# def render(rgb):
#     if MODES[mode] == "rgb":
#         full = rgb
#     elif MODES[mode] == "depth":
#         full = colorize(depth)
#     else:
#         full = cv2.addWeighted(rgb, 0.5, colorize(depth), 0.5, 0)

#     cw, ch = crop_size()
#     crop = full[y0:y0 + ch, x0:x0 + cw]
#     interp = cv2.INTER_NEAREST if zoom >= 2 else cv2.INTER_LINEAR
#     canvas = cv2.resize(crop, (rw, rh), interpolation=interp)

#     # measured points and the line between them
#     pts_c = [img_to_canvas(p["u"], p["v"]) for p in points]
#     if len(pts_c) == 2:
#         cv2.line(canvas, pts_c[0], pts_c[1], (0, 255, 255), 2)
#     for p, (px, py) in zip(points, pts_c):
#         cv2.drawMarker(canvas, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
#         cv2.putText(canvas, f"Z={p['Z']:.3f}m", (px + 12, py - 12),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     status = f"frame {frame_idx}  [{MODES[mode]}]  zoom {zoom:.1f}x"
#     if paused:
#         status += "  PAUSED"
#     cv2.putText(canvas, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
#     cv2.putText(canvas, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
#     cv2.imshow("rgbd", canvas)


# # -------------------------------------------------------------------- main --
# cv2.namedWindow("rgbd", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("rgbd", 1280, 720)
# cv2.setMouseCallback("rgbd", on_mouse)

# paused = False
# quit_now = False
# while not quit_now:
#     rbuf = rgb_proc.stdout.read(RGB_BYTES)
#     dbuf = dep_proc.stdout.read(DEP_BYTES)
#     if len(rbuf) < RGB_BYTES or len(dbuf) < DEP_BYTES:
#         print(f"End of video after {frame_idx} frames "
#               f"({'RGB' if len(rbuf) < RGB_BYTES else 'depth'} ended first).")
#         break

#     rgb = np.frombuffer(rbuf, np.uint8).reshape(rh, rw, 3)
#     d = np.frombuffer(dbuf, np.uint16).reshape(dh, dw)
#     if (dw, dh) != (rw, rh):
#         d = cv2.resize(d, (rw, rh), interpolation=cv2.INTER_NEAREST)
#     depth = d

#     if frame_idx == 0:
#         valid = depth[depth > 0]
#         if valid.size:
#             print(f"Frame 0 depth raw -> min {valid.min()}, max {valid.max()}, "
#                   f"median {np.median(valid):.0f}, valid {valid.size / depth.size:.0%}")
#         print("Click to measure, wheel to zoom, right-drag to pan. "
#               "Any other key starts playback.")

#     while True:   # redraw loop: keeps zoom/clicks live while paused
#         render(rgb)
#         hold = frame_idx == 0 or paused
#         key = cv2.waitKey(30) & 0xFF
#         if cv2.getWindowProperty("rgbd", cv2.WND_PROP_VISIBLE) < 1:
#             quit_now = True
#             break
#         if key == ord("q"):
#             quit_now = True
#             break
#         if key == ord("m"):
#             mode = (mode + 1) % len(MODES)
#             continue
#         if key in (ord("+"), ord("=")):
#             zoom_at(rw // 2, rh // 2, 1.25); continue
#         if key == ord("-"):
#             zoom_at(rw // 2, rh // 2, 0.8); continue
#         if key == ord("r"):
#             zoom, x0, y0 = 1.0, 0, 0; continue
#         if key == ord("c"):
#             points.clear(); print("points cleared"); continue
#         if key == ord(" "):
#             paused = not paused
#             if paused:
#                 continue
#             break
#         if hold and key == 255:   # waiting on frame 0 / paused, nothing pressed
#             continue
#         if hold and key != 255:   # any other key on frame 0 starts playback
#             break
#         break                     # playing: advance to next frame
#     frame_idx += 1

# rgb_proc.terminate()
# dep_proc.terminate()
# cv2.destroyAllWindows()





"""
RGB + depth pair viewer for OAK recordings.
Click on the image -> X, Y, Z (metres) at that pixel, plus 3D distance between
the last two clicks. Zoom and pan supported.

Needs only ffmpeg (no ffprobe), numpy and opencv-python (NOT the headless build).

Usage:
    python rgbd_viewer.py
    python rgbd_viewer.py <rgb_video> <depth_video>

Mouse:
    Left-click        : measure X, Y, Z at that pixel (2 clicks -> 3D distance)
    Mouse wheel       : zoom in / out at the cursor
    Right-button drag : pan when zoomed in
Keys:
    + / -   : zoom in / out (centre of view)
    r       : reset zoom
    c       : clear measured points
    m       : switch view  RGB -> depth -> overlay
    Space   : pause / resume (clicking and zooming work while paused)
    q       : quit
"""
import re
import subprocess
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------- settings --
RGB_PATH = r"C:\Users\EmageVision\Documents\head_training_dataset\videos\tomo_socketing_video\raw and depth\20261027_145529_648_video.mp4"
DEPTH_PATH = r"C:\Users\EmageVision\Documents\head_training_dataset\videos\tomo_socketing_video\raw and depth\20261027_145529_648_depth.mkv"
DEPTH_SCALE = 0.001       # OAK depth is uint16 millimetres -> metres
NEAR, FAR = 200, 3000     # colour range for depth display only (mm)

# OAK RGB (CAM_A) intrinsics from the camera calibration, valid at 1920x1080
FX, FY = 1070.789795, 1070.526978
CX, CY = 962.309387, 550.0
CALIB_W, CALIB_H = 1920, 1080

if len(sys.argv) == 3:
    RGB_PATH, DEPTH_PATH = sys.argv[1], sys.argv[2]


# ------------------------------------------------------------------ probing --
def probe(path):
    """Return (width, height, fps, stream_line, duration) using ffmpeg -i."""
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", path],
                         capture_output=True, text=True).stderr
    lines = [l.strip() for l in out.splitlines() if "Video:" in l]
    if not lines:
        print(out)
        sys.exit(f"No video stream found in {path}")
    line = lines[0]
    w, h = map(int, re.search(r"\b(\d{2,5})x(\d{2,5})\b", line).groups())
    fps_m = re.search(r"([\d.]+) fps", line)
    fps = float(fps_m.group(1)) if fps_m else None
    dur_m = re.search(r"Duration: ([\d:.]+)", out)
    return w, h, fps, line, (dur_m.group(1) if dur_m else "?")


rw, rh, rfps, rline, rdur = probe(RGB_PATH)
dw, dh, dfps, dline, ddur = probe(DEPTH_PATH)
print("RGB   stream:", rline, "| duration", rdur)
print("Depth stream:", dline, "| duration", ddur)

if not any(f in dline for f in ("gray16", "y16", "b16g", "gray12", "gray10")):
    print("\n*** WARNING: depth stream is not 16-bit grayscale. If it is "
          "h264/hevc/yuv420p, the real distances are lost. ***\n")
if (rw, rh) != (dw, dh):
    print(f"\nNOTE: RGB is {rw}x{rh} but depth is {dw}x{dh}. Depth will be "
          "resized to the RGB size; only correct if depth was aligned to RGB.\n")
if rfps and dfps and abs(rfps - dfps) > 0.5:
    print(f"\nNOTE: fps differ (RGB {rfps}, depth {dfps}); frames may drift.\n")

# Intrinsics are for 1920x1080; rescale if the RGB video is a different size
sx, sy = rw / CALIB_W, rh / CALIB_H
fx, fy, cx, cy = FX * sx, FY * sy, CX * sx, CY * sy
if (rw, rh) != (CALIB_W, CALIB_H):
    print(f"NOTE: intrinsics rescaled from {CALIB_W}x{CALIB_H} to {rw}x{rh}.")
print(f"Intrinsics used: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")


# ----------------------------------------------------------------- readers --
def open_pipe(path, pix_fmt):
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0",
         "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
        stdout=subprocess.PIPE)


rgb_proc = open_pipe(RGB_PATH, "bgr24")
dep_proc = open_pipe(DEPTH_PATH, "gray16le")
RGB_BYTES = rw * rh * 3
DEP_BYTES = dw * dh * 2

# ------------------------------------------------------------------- state --
depth = None              # current depth frame (uint16, RGB size)
frame_idx = 0
MODES = ["rgb", "depth", "overlay"]
mode = 0
points = []               # measured points: dict(u, v, X, Y, Z, frame)

zoom = 1.0                # 1 = full image
x0, y0 = 0, 0             # top-left of the visible crop, in image pixels
drag = None               # (mouse_x, mouse_y, x0, y0) at right-button press
MAX_ZOOM = 20.0


def crop_size():
    return max(1, int(round(rw / zoom))), max(1, int(round(rh / zoom)))


def clamp_view():
    global x0, y0
    cw, ch = crop_size()
    x0 = int(round(min(max(x0, 0), rw - cw)))
    y0 = int(round(min(max(y0, 0), rh - ch)))


def canvas_to_img(x, y):
    """Mouse position on the displayed canvas -> pixel in the full image."""
    cw, ch = crop_size()
    u = x0 + x * cw / rw
    v = y0 + y * ch / rh
    return int(min(max(u, 0), rw - 1)), int(min(max(v, 0), rh - 1))


def img_to_canvas(u, v):
    cw, ch = crop_size()
    return int((u - x0 + 0.5) * rw / cw), int((v - y0 + 0.5) * rh / ch)


def zoom_at(x, y, factor):
    """Zoom keeping the image point under canvas (x, y) fixed."""
    global zoom, x0, y0
    cw, ch = crop_size()
    pu = x0 + x * cw / rw
    pv = y0 + y * ch / rh
    zoom = float(min(max(zoom * factor, 1.0), MAX_ZOOM))
    cw, ch = crop_size()
    x0 = pu - x * cw / rw
    y0 = pv - y * ch / rh
    clamp_view()


def measure(u, v):
    patch = depth[max(v - 2, 0):v + 3, max(u - 2, 0):u + 3].astype(np.float32)
    patch = patch[patch > 0]
    if not patch.size:
        print(f"frame {frame_idx}  pixel ({u},{v}) -> no depth (0)")
        return
    Z = float(np.median(patch)) * DEPTH_SCALE
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    points.append(dict(u=u, v=v, X=X, Y=Y, Z=Z, frame=frame_idx))
    del points[:-2]       # keep the last two for the distance measurement
    print(f"frame {frame_idx}  pixel ({u},{v}) -> "
          f"X={X:+.3f}  Y={Y:+.3f}  Z={Z:.3f} m")
    if len(points) == 2:
        a, b = points
        dist = np.sqrt((a["X"] - b["X"])**2 + (a["Y"] - b["Y"])**2 + (a["Z"] - b["Z"])**2)
        print(f"   3D distance between last two points: {dist * 1000:.1f} mm")


def on_mouse(event, x, y, flags, param):
    global drag, x0, y0
    if depth is None:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        measure(*canvas_to_img(x, y))
    elif event == cv2.EVENT_MOUSEWHEEL:
        zoom_at(x, y, 1.25 if cv2.getMouseWheelDelta(flags) > 0 else 0.8)
    elif event == cv2.EVENT_RBUTTONDOWN:
        drag = (x, y, x0, y0)
    elif event == cv2.EVENT_MOUSEMOVE and drag is not None and (flags & cv2.EVENT_FLAG_RBUTTON):
        cw, ch = crop_size()
        x0 = drag[2] - (x - drag[0]) * cw / rw
        y0 = drag[3] - (y - drag[1]) * ch / rh
        clamp_view()
    elif event == cv2.EVENT_RBUTTONUP:
        drag = None


def colorize(d):
    vis = np.clip((d.astype(np.float32) - NEAR) / (FAR - NEAR), 0, 1)
    vis = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis[d == 0] = 0
    return vis


def render(rgb):
    if MODES[mode] == "rgb":
        full = rgb
    elif MODES[mode] == "depth":
        full = colorize(depth)
    else:
        full = cv2.addWeighted(rgb, 0.5, colorize(depth), 0.5, 0)

    cw, ch = crop_size()
    crop = full[y0:y0 + ch, x0:x0 + cw]
    interp = cv2.INTER_NEAREST if zoom >= 2 else cv2.INTER_LINEAR
    canvas = cv2.resize(crop, (rw, rh), interpolation=interp)

    # measured points and the line between them
    pts_c = [img_to_canvas(p["u"], p["v"]) for p in points]
    if len(pts_c) == 2:
        cv2.line(canvas, pts_c[0], pts_c[1], (0, 255, 255), 2)
    for p, (px, py) in zip(points, pts_c):
        cv2.drawMarker(canvas, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
        cv2.putText(canvas, f"Z={p['Z']:.3f}m", (px + 12, py - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    status = f"frame {frame_idx}  [{MODES[mode]}]  zoom {zoom:.1f}x"
    if paused:
        status += "  PAUSED"
    cv2.putText(canvas, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
    cv2.putText(canvas, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.imshow("rgbd", canvas)


# -------------------------------------------------------------------- main --
cv2.namedWindow("rgbd", cv2.WINDOW_NORMAL)
cv2.resizeWindow("rgbd", 1280, 720)
cv2.setMouseCallback("rgbd", on_mouse)

paused = False
quit_now = False
while not quit_now:
    rbuf = rgb_proc.stdout.read(RGB_BYTES)
    dbuf = dep_proc.stdout.read(DEP_BYTES)
    if len(rbuf) < RGB_BYTES or len(dbuf) < DEP_BYTES:
        print(f"End of video after {frame_idx} frames "
              f"({'RGB' if len(rbuf) < RGB_BYTES else 'depth'} ended first).")
        break

    rgb = np.frombuffer(rbuf, np.uint8).reshape(rh, rw, 3)
    d = np.frombuffer(dbuf, np.uint16).reshape(dh, dw)
    if (dw, dh) != (rw, rh):
        d = cv2.resize(d, (rw, rh), interpolation=cv2.INTER_NEAREST)
    depth = d

    if frame_idx == 0:
        valid = depth[depth > 0]
        if valid.size:
            print(f"Frame 0 depth raw -> min {valid.min()}, max {valid.max()}, "
                  f"median {np.median(valid):.0f}, valid {valid.size / depth.size:.0%}")
        print("Click to measure, wheel to zoom, right-drag to pan. "
              "Any other key starts playback.")

    while True:   # redraw loop: keeps zoom/clicks live while paused
        render(rgb)
        hold = frame_idx == 0 or paused
        key = cv2.waitKey(30) & 0xFF
        if cv2.getWindowProperty("rgbd", cv2.WND_PROP_VISIBLE) < 1:
            quit_now = True
            break
        if key == ord("q"):
            quit_now = True
            break
        if key == ord("m"):
            mode = (mode + 1) % len(MODES)
            continue
        if key in (ord("+"), ord("=")):
            zoom_at(rw // 2, rh // 2, 1.25); continue
        if key == ord("-"):
            zoom_at(rw // 2, rh // 2, 0.8); continue
        if key == ord("r"):
            zoom, x0, y0 = 1.0, 0, 0; continue
        if key == ord("c"):
            points.clear(); print("points cleared"); continue
        if key == ord(" "):
            paused = not paused
            if paused:
                continue
            break
        if hold and key == 255:   # waiting on frame 0 / paused, nothing pressed
            continue
        if hold and key != 255:   # any other key on frame 0 starts playback
            break
        break                     # playing: advance to next frame
    frame_idx += 1

rgb_proc.terminate()
dep_proc.terminate()
cv2.destroyAllWindows()