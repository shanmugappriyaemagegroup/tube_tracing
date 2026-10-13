"""
select_roi.py - draw a region of interest (ROI) on one frame of your video by
clicking points, so run_video_frames.py can ignore anything detected outside
it (a watch on your wrist near the frame edge, clutter on a shelf, etc).
Nothing is passed on the command line - edit CONFIG below and run:

    python select_roi.py

A polygon (not just a rectangle) is used because your table is shown at an
angle in the video - a plain rectangle would either cut off part of the real
work area or let in more background than needed. Click as many points as you
like to trace the actual usable area.

Controls (shown in the window title bar too):
  left click   add a point to the polygon
  u            undo the last point
  c            close the polygon (needs at least 3 points) and preview the fill
  s            save to ROI_JSON and quit
  r            reset and start over
  q            quit without saving
"""
import json
from pathlib import Path

import cv2
import numpy as np

# ============================== CONFIG — edit these ==============================
VIDEO_PATH = r"C:\Users\EmageVision\Documents\head_tail_classification\videos\part_0.avi"
FRAME_NUMBER = 0        # which frame to draw on - pick one where the full work
                       # area and any usual clutter (like your watch) are visible
ROI_JSON = r"C:\Users\EmageVision\Documents\head_tail_classification\roi.json"
# ===================================================================================


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {VIDEO_PATH}")
    for _ in range(FRAME_NUMBER + 1):
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {FRAME_NUMBER} from {VIDEO_PATH}")
    h, w = frame.shape[:2]

    points = []
    closed = [False]

    def redraw():
        vis = frame.copy()
        for p in points:
            cv2.circle(vis, p, 5, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.polylines(vis, [np.array(points)], closed[0], (0, 255, 0), 2)
        if closed[0] and len(points) >= 3:
            overlay = vis.copy()
            cv2.fillPoly(overlay, [np.array(points)], (0, 255, 0))
            vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
        cv2.putText(vis, "click=add point  u=undo  c=close  s=save  r=reset  q=quit",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("select ROI", vis)

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and not closed[0]:
            points.append((x, y))
            redraw()

    cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("select ROI", on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            print("[quit] nothing saved")
            break
        if k == ord("u") and points and not closed[0]:
            points.pop()
            redraw()
        if k == ord("r"):
            points.clear()
            closed[0] = False
            redraw()
        if k == ord("c"):
            if len(points) < 3:
                print("[info] need at least 3 points to close a polygon")
            else:
                closed[0] = True
                redraw()
        if k == ord("s"):
            if len(points) < 3 or not closed[0]:
                print("[info] close the polygon with 'c' first (need >= 3 points)")
                continue
            Path(ROI_JSON).parent.mkdir(parents=True, exist_ok=True)
            Path(ROI_JSON).write_text(json.dumps({
                "points": points,   # not tied to VIDEO_PATH - reusable for any video from
                                    # the same camera position/framing; redraw only if that changes
            }, indent=2))
            print(f"[saved] {ROI_JSON}  ({len(points)} points)")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()