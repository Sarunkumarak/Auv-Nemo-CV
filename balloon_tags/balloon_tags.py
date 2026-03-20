import cv2
import cv2.aruco as aruco
from pupil_apriltags import Detector
import numpy as np
import threading
import queue
import time

# ─────────────────────────────────────────────
#  CAMERA
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))

if not cap.isOpened():
    print("[ERROR] Camera failed to open")
    exit(1)
else:
    print(f"[INFO] Camera opened — "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

# ─────────────────────────────────────────────
#  ARUCO SETUP
# ─────────────────────────────────────────────
ARUCO_DICTS         = [aruco.DICT_4X4_1000, aruco.DICT_5X5_1000,
                       aruco.DICT_6X6_1000, aruco.DICT_7X7_1000]
ARUCO_DICT_NAMES    = ["4x4", "5x5", "6x6", "7x7"]
ARUCO_DICT_PRIORITY = [1, 2, 3, 4]

aruco_params = aruco.DetectorParameters()
aruco_params.minMarkerPerimeterRate      = 0.03
aruco_params.maxMarkerPerimeterRate      = 10.0
aruco_params.adaptiveThreshWinSizeMin    = 3
aruco_params.adaptiveThreshWinSizeMax    = 53
aruco_params.adaptiveThreshWinSizeStep   = 4
aruco_params.adaptiveThreshConstant      = 7
aruco_params.minCornerDistanceRate       = 0.05
aruco_params.polygonalApproxAccuracyRate = 0.03
aruco_params.errorCorrectionRate         = 0.6
aruco_params.minMarkerDistanceRate       = 0.05
aruco_params.cornerRefinementMethod      = aruco.CORNER_REFINE_SUBPIX

aruco_detectors = [
    aruco.ArucoDetector(aruco.getPredefinedDictionary(d), aruco_params)
    for d in ARUCO_DICTS
]

# ─────────────────────────────────────────────
#  APRILTAG
# ─────────────────────────────────────────────
apriltag_detector = Detector(families="tag36h11")

# ─────────────────────────────────────────────
#  QR
# ─────────────────────────────────────────────
try:
    qr_detector = cv2.QRCodeDetectorAruco()
    print("[INFO] QRCodeDetectorAruco loaded")
except AttributeError:
    qr_detector = cv2.QRCodeDetector()
    print("[INFO] QRCodeDetector (fallback) loaded")

# ─────────────────────────────────────────────
#  SHARED ENHANCEMENT
# ─────────────────────────────────────────────
clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
gamma_lut = np.array([((i / 255.0) ** (1.0 / 1.8)) * 255
                      for i in range(256)], dtype=np.uint8)
SHARPEN_KERNEL = np.array([[ 0, -1,  0],
                            [-1,  5, -1],
                            [ 0, -1,  0]], dtype=np.float32)

# ─────────────────────────────────────────────
#  BALLOON CONFIG
# ─────────────────────────────────────────────
BALLOON_AREA_THRESHOLD  = 3000
BALLOON_DIAMETER_M      = 0.25
BALLOON_RESULT_TTL      = 0.30
BALLOON_SCALE           = 0.5
BALLOON_EARLY_EXIT_AREA = 8000

COLOR_RANGES = {
    "RED": [
        (np.array([0,   100,  70]), np.array([8,   255, 255])),
        (np.array([170, 100,  70]), np.array([180, 255, 255]))
    ],
    "ORANGE": [(np.array([8,  100, 80]), np.array([20,  255, 255]))],
    "YELLOW": [(np.array([23, 100, 80]), np.array([35,  255, 255]))],
    "GREEN":  [(np.array([35,  80, 60]), np.array([85,  255, 255]))],
    "BLUE":   [(np.array([95, 100, 60]), np.array([140, 255, 255]))]
}

# ─────────────────────────────────────────────
#  TUNABLE CONSTANTS
# ─────────────────────────────────────────────
CENTER_TOLERANCE    = 60
MAX_LOST            = 8        # reduced — drop stale lock faster
NIGHT_THRESHOLD     = 80
MIN_MARKER_AREA     = 500      # reduced for half-res
MIN_QR_AREA         = 1500
QR_RESULT_TTL       = 0.25     # tighter
MARKER_RESULT_TTL   = 0.10     # tighter — stale results drop fast
SPATIAL_MERGE_DIST  = 80
ALIGNMENT_THRESHOLD = 30
DETECT_SCALE        = 0.5      # ArUco + April run on half-res too

# ─────────────────────────────────────────────
#  SHARED DISPLAY FRAMES
# ─────────────────────────────────────────────
display_lock         = threading.Lock()
shared_balloon_mask  = None
shared_balloon_edges = None

# ─────────────────────────────────────────────
#  MUTABLE GLOBALS
# ─────────────────────────────────────────────
last_qr_result      = None
last_qr_ts          = 0.0
last_balloon_result = None
last_balloon_ts     = 0.0
last_marker_result  = None
last_marker_ts      = 0.0
last_target         = None
lost_frames         = 0

# ─────────────────────────────────────────────
#  KEY HELPER — flush + put (always latest frame)
# ─────────────────────────────────────────────
def force_put(q, item):
    """Drain queue then put — thread always gets the LATEST frame."""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break
    try:
        q.put_nowait(item)
    except queue.Full:
        pass

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def is_valid_marker(pts, scale=1.0):
    area = cv2.contourArea(pts.astype(np.float32))
    if area < MIN_MARKER_AREA * (scale ** 2):
        return False
    sides = [np.linalg.norm(pts[(i+1) % 4] - pts[i]) for i in range(4)]
    mean  = np.mean(sides)
    if mean == 0:
        return False
    if np.any(np.abs(sides - mean) / mean > 0.40):
        return False
    rect = cv2.minAreaRect(pts.astype(np.float32))
    rw, rh = rect[1]
    if min(rw, rh) == 0:
        return False
    if max(rw, rh) / min(rw, rh) > 1.6:
        return False
    return True

def is_balloon_contour(cnt, min_circularity=0.65):
    area  = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    if perim == 0:
        return False
    return (4 * np.pi * area / (perim ** 2)) > min_circularity

# ─────────────────────────────────────────────
#  ARUCO THREADS — one per dict, half-res input
# ─────────────────────────────────────────────
aruco_input_queues = [queue.Queue(maxsize=2) for _ in ARUCO_DICTS]
aruco_result_queue = queue.Queue(maxsize=8)

def make_aruco_worker(det_idx):
    detector  = aruco_detectors[det_idx]
    priority  = ARUCO_DICT_PRIORITY[det_idx]
    dict_name = ARUCO_DICT_NAMES[det_idx]
    in_q      = aruco_input_queues[det_idx]

    def worker():
        while True:
            try:
                gray_s, enhanced_s, scale = in_q.get(timeout=1.0)
            except queue.Empty:
                continue

            corners, ids, _ = detector.detectMarkers(enhanced_s)
            if ids is None:
                corners, ids, _ = detector.detectMarkers(gray_s)
            if ids is None or len(ids) == 0:
                continue

            ts = time.monotonic()
            for i in range(len(ids)):
                c = corners[i][0]
                if not is_valid_marker(c, scale):
                    continue
                area = cv2.contourArea(c)
                # Scale coords back to full res
                tx   = int(c[:, 0].mean() / scale)
                ty   = int(c[:, 1].mean() / scale)
                corner_full = (corners[i] / scale).astype(np.float32)
                result = {
                    "type":     "ARUCO",
                    "priority": priority,
                    "area":     area,
                    "tx": tx,   "ty": ty,
                    "label":    f"ArUco {dict_name}:{int(ids[i][0])}",
                    "source":   f"ARUCO-{dict_name}",
                    "aruco_id": int(ids[i][0]),
                    "corner":   corner_full,
                    "aid":      ids[i].copy(),
                }
                try:
                    aruco_result_queue.put_nowait((result, ts))
                except queue.Full:
                    pass
    return worker

# ─────────────────────────────────────────────
#  APRILTAG THREAD — half-res input
# ─────────────────────────────────────────────
april_input_queue  = queue.Queue(maxsize=2)
april_result_queue = queue.Queue(maxsize=2)

def april_worker():
    while True:
        try:
            enhanced_s, scale = april_input_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        tags   = apriltag_detector.detect(enhanced_s)
        result = None
        for tag in tags:
            tx = int(tag.center[0] / scale)
            ty = int(tag.center[1] / scale)
            result = {
                "type":   "APRIL",
                "tx": tx, "ty": ty,
                "label":  f"April {tag.tag_id}",
                "source": "APRIL",
            }
            print(f"[AprilTag] ID:{tag.tag_id}  X:{tx}  Y:{ty}")
            break

        # Always push (flush old first)
        force_put(april_result_queue, (result, time.monotonic()))

# ─────────────────────────────────────────────
#  QR THREAD
# ─────────────────────────────────────────────
qr_input_queue  = queue.Queue(maxsize=2)
qr_result_queue = queue.Queue(maxsize=2)

def preprocess_for_qr(gray):
    sharp = cv2.filter2D(gray, -1, SHARPEN_KERNEL)
    return cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)

def try_detect_qr(img, cx, cy):
    result = None
    try:
        ok, data_list, pts_list, _ = qr_detector.detectAndDecodeMulti(img)
        if ok and data_list:
            best_dist, best_i = float('inf'), -1
            for i, (data, pts) in enumerate(zip(data_list, pts_list)):
                if not data or pts is None:
                    continue
                area = cv2.contourArea(pts.astype(np.float32))
                if area < MIN_QR_AREA:
                    continue
                qx   = pts[:, 0].mean()
                qy   = pts[:, 1].mean()
                dist = (qx - cx) ** 2 + (qy - cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_i    = i
            if best_i >= 0:
                pts    = pts_list[best_i].astype(int)
                result = (int(pts[:, 0].mean()), int(pts[:, 1].mean()),
                          data_list[best_i], pts)
    except Exception:
        try:
            data, pts, _ = qr_detector.detectAndDecode(img)
            if data and pts is not None:
                pts = pts[0].astype(int)
                if cv2.contourArea(pts.astype(np.float32)) >= MIN_QR_AREA:
                    result = (int(pts[:, 0].mean()), int(pts[:, 1].mean()),
                              data, pts)
        except Exception:
            pass
    return result

def qr_worker():
    while True:
        try:
            gray, cx, cy = qr_input_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        result = try_detect_qr(gray, cx, cy)
        if result is None:
            result = try_detect_qr(
                cv2.filter2D(gray, -1, SHARPEN_KERNEL), cx, cy)
        if result is None:
            result = try_detect_qr(clahe.apply(gray), cx, cy)
        if result is None:
            result = try_detect_qr(preprocess_for_qr(gray), cx, cy)
        if result is None:
            small = cv2.resize(gray, (640, 360))
            r     = try_detect_qr(small, cx // 2, cy // 2)
            if r is not None:
                tx, ty, data, pts = r
                result = (tx * 2, ty * 2, data, (pts * 2).astype(int))
        force_put(qr_result_queue, (result, time.monotonic()))

# ─────────────────────────────────────────────
#  BALLOON THREAD
# ─────────────────────────────────────────────
balloon_input_queue  = queue.Queue(maxsize=2)
balloon_result_queue = queue.Queue(maxsize=2)

def balloon_worker():
    global shared_balloon_mask, shared_balloon_edges

    while True:
        try:
            frame_small, scale, full_w, full_h = \
                balloon_input_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        blurred = cv2.GaussianBlur(frame_small, (5, 5), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray_s  = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        edges_s = cv2.Canny(gray_s, 50, 150)

        best_contour   = None
        detected_color = None
        max_area       = 0
        best_mask_s    = None
        early_exit     = False

        for color_name, ranges in COLOR_RANGES.items():
            if early_exit:
                break
            mask = None
            for lower, upper in ranges:
                temp = cv2.inRange(hsv, lower, upper)
                mask = temp if mask is None else cv2.bitwise_or(mask, temp)

            mask = cv2.medianBlur(mask, 5)
            mask = cv2.erode(mask,  None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=1)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < BALLOON_AREA_THRESHOLD * (scale ** 2):
                    continue
                if not is_balloon_contour(cnt):
                    continue
                contour_mask = np.zeros_like(gray_s)
                cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
                edge_px = cv2.countNonZero(
                    cv2.bitwise_and(edges_s, edges_s, mask=contour_mask))
                if edge_px < 30:
                    continue
                if area > max_area:
                    max_area       = area
                    best_contour   = cnt
                    detected_color = color_name
                    best_mask_s    = mask.copy()
                    if area > BALLOON_EARLY_EXIT_AREA * (scale ** 2):
                        early_exit = True
                        break

        mask_full = cv2.resize(
            best_mask_s if best_mask_s is not None
            else np.zeros(
                (frame_small.shape[0], frame_small.shape[1]), dtype=np.uint8),
            (full_w, full_h), interpolation=cv2.INTER_NEAREST)
        edges_full = cv2.resize(
            edges_s, (full_w, full_h), interpolation=cv2.INTER_NEAREST)

        with display_lock:
            shared_balloon_mask  = mask_full
            shared_balloon_edges = edges_full

        result = None
        if best_contour is not None:
            (x, y), radius = cv2.minEnclosingCircle(best_contour)
            result = (
                (int(x / scale), int(y / scale)),
                radius / scale,
                detected_color,
                (best_contour.astype(np.float32) / scale).astype(np.int32)
            )

        force_put(balloon_result_queue, (result, time.monotonic()))

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    global last_qr_result,      last_qr_ts
    global last_balloon_result, last_balloon_ts
    global last_marker_result,  last_marker_ts
    global last_target, lost_frames

    # Start all threads
    for i in range(len(ARUCO_DICTS)):
        threading.Thread(target=make_aruco_worker(i), daemon=True).start()
    threading.Thread(target=april_worker,   daemon=True).start()
    threading.Thread(target=qr_worker,      daemon=True).start()
    threading.Thread(target=balloon_worker, daemon=True).start()

    print("[INFO] Integrated Detection — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        h, w   = frame.shape[:2]
        cx, cy = w // 2, h // 2
        now    = time.monotonic()

        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = cv2.mean(gray)[0]

        if brightness < NIGHT_THRESHOLD:
            enhanced = clahe.apply(cv2.LUT(gray, gamma_lut))
            mode     = "NIGHT"
        else:
            enhanced = cv2.equalizeHist(gray)
            mode     = "DAY"

        # ── Half-res for marker detection ──
        gray_s     = cv2.resize(gray,     (0, 0),
                                fx=DETECT_SCALE, fy=DETECT_SCALE)
        enhanced_s = cv2.resize(enhanced, (0, 0),
                                fx=DETECT_SCALE, fy=DETECT_SCALE)

        # ════════════════════════════════════════════════════════════
        #  FEED ALL THREADS — force_put = always latest frame, no stale
        # ════════════════════════════════════════════════════════════
        for q in aruco_input_queues:
            force_put(q, (gray_s, enhanced_s, DETECT_SCALE))

        force_put(april_input_queue,   (enhanced_s, DETECT_SCALE))
        force_put(qr_input_queue,      (gray.copy(), cx, cy))
        force_put(balloon_input_queue, (
            cv2.resize(frame, (0, 0), fx=BALLOON_SCALE, fy=BALLOON_SCALE),
            BALLOON_SCALE, w, h))

        # ════════════════════════════════════════════════════════════
        #  COLLECT RESULTS
        # ════════════════════════════════════════════════════════════

        # Drain ALL fresh ArUco results, pick best
        fresh_aruco = []
        while True:
            try:
                r, ts = aruco_result_queue.get_nowait()
                if (now - ts) < MARKER_RESULT_TTL:
                    fresh_aruco.append((r, ts))
            except queue.Empty:
                break

        if fresh_aruco:
            all_det = [(r["priority"], r["area"],
                        r["tx"], r["ty"], r) for r, _ in fresh_aruco]
            used   = [False] * len(all_det)
            groups = []
            for i in range(len(all_det)):
                if used[i]:
                    continue
                group   = [i]
                used[i] = True
                for j in range(i + 1, len(all_det)):
                    if used[j]:
                        continue
                    dist = ((all_det[i][2] - all_det[j][2]) ** 2 +
                            (all_det[i][3] - all_det[j][3]) ** 2) ** 0.5
                    if dist < SPATIAL_MERGE_DIST:
                        group.append(j)
                        used[j] = True
                groups.append(group)

            best_r, best_score = None, (-1, -1)
            for group in groups:
                for idx in group:
                    pri, area = all_det[idx][0], all_det[idx][1]
                    if (pri, area) > best_score:
                        best_score = (pri, area)
                        best_r     = all_det[idx][4]

            if best_r:
                last_marker_result = best_r
                last_marker_ts     = now
                print(f"[ArUco-{best_r['source'].split('-')[1]}] "
                      f"ID:{best_r['aruco_id']}  "
                      f"X:{best_r['tx']}  Y:{best_r['ty']}")

        # AprilTag
        try:
            ap_res, ap_ts = april_result_queue.get_nowait()
            if ap_res is not None:
                if (last_marker_result is None or
                        last_marker_result["type"] != "ARUCO" or
                        (now - last_marker_ts) >= MARKER_RESULT_TTL):
                    last_marker_result = ap_res
                    last_marker_ts     = ap_ts
        except queue.Empty:
            pass

        # QR
        try:
            qr_res, qr_ts  = qr_result_queue.get_nowait()
            last_qr_result = qr_res
            last_qr_ts     = qr_ts
        except queue.Empty:
            pass

        # Balloon
        try:
            b_res, b_ts         = balloon_result_queue.get_nowait()
            last_balloon_result = b_res
            last_balloon_ts     = b_ts
        except queue.Empty:
            pass

        # ════════════════════════════════════════════════════════════
        #  PRIORITY: Marker → QR → Balloon
        # ════════════════════════════════════════════════════════════
        detected      = None
        detect_source = ""

        # 1. Marker
        if (last_marker_result is not None and
                (now - last_marker_ts) < MARKER_RESULT_TTL):
            r  = last_marker_result
            tx, ty = r["tx"], r["ty"]
            detected      = (tx, ty, r["label"])
            detect_source = r["source"]
            if r["type"] == "ARUCO":
                aruco.drawDetectedMarkers(
                    frame, [r["corner"]], np.array([[r["aruco_id"]]]))
        else:
            if (now - last_marker_ts) >= MARKER_RESULT_TTL:
                last_marker_result = None

        # 2. QR
        if detected is None:
            if (last_qr_result is not None and
                    (now - last_qr_ts) < QR_RESULT_TTL):
                tx, ty, data, pts = last_qr_result
                detected      = (tx, ty, f"QR:{data}")
                detect_source = "QR"
                for i in range(len(pts)):
                    cv2.line(frame, tuple(pts[i]),
                             tuple(pts[(i + 1) % len(pts)]),
                             (0, 255, 0), 2)
                cv2.putText(frame, f"{data[:30]}", (tx - 20, ty + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 0), 2)
                print(f"[QR] Data:{data}  X:{tx}  Y:{ty}")
            else:
                if (now - last_qr_ts) >= QR_RESULT_TTL:
                    last_qr_result = None

        # 3. Balloon
        if detected is None:
            if (last_balloon_result is not None and
                    (now - last_balloon_ts) < BALLOON_RESULT_TTL):
                b_center, b_radius, b_color, b_contour = last_balloon_result
                bx, by  = b_center
                dx_px   = bx - cx
                dy_px   = cy - by
                mpp     = BALLOON_DIAMETER_M / (2 * b_radius + 1e-6)
                dx_m    = dx_px * mpp
                dy_m    = dy_px * mpp
                dist_m  = np.sqrt(dx_m ** 2 + dy_m ** 2)
                aligned = (abs(dx_px) <= ALIGNMENT_THRESHOLD and
                           abs(dy_px) <= ALIGNMENT_THRESHOLD)

                detected      = (bx, by, f"Balloon:{b_color}")
                detect_source = "BALLOON"
                print(f"[Balloon] {b_color}  X:{bx}  Y:{by}")

                cv2.drawContours(frame, [b_contour], -1, (0, 255, 0), 2)
                cv2.circle(frame, b_center, int(b_radius), (255, 0, 0), 2)
                cv2.circle(frame, b_center, 5, (0, 0, 255), -1)
                cv2.line(frame, (cx, cy), b_center, (255, 255, 0), 2)

                cv2.putText(frame, f"BALLOON: {b_color}", (20, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(frame, f"Offset : {dist_m:.2f} m", (20, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                if aligned:
                    cv2.putText(frame, "ALIGNED", (20, 155),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                                (0, 255, 0), 2)
                else:
                    moves = []
                    if abs(dx_px) > ALIGNMENT_THRESHOLD:
                        moves.append(
                            f"{'RIGHT' if dx_px > 0 else 'LEFT'}"
                            f" {abs(dx_m):.2f}m")
                    if abs(dy_px) > ALIGNMENT_THRESHOLD:
                        moves.append(
                            f"{'UP' if dy_px > 0 else 'DOWN'}"
                            f" {abs(dy_m):.2f}m")
                    cv2.putText(frame, "MOVE: " + " | ".join(moves),
                                (20, 155), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 255), 2)
            else:
                if (now - last_balloon_ts) >= BALLOON_RESULT_TTL:
                    last_balloon_result = None

        # ════════════════════════════════════════════════════════════
        #  STABILITY FILTER
        # ════════════════════════════════════════════════════════════
        if detected is not None:
            last_target = detected
            lost_frames = 0
        else:
            lost_frames += 1
            if lost_frames >= MAX_LOST:
                last_target         = None
                last_qr_result      = None
                last_balloon_result = None
                last_marker_result  = None

        # ════════════════════════════════════════════════════════════
        #  DISPLAY
        # ════════════════════════════════════════════════════════════
        if last_target is not None and lost_frames < MAX_LOST:
            tx, ty, label = last_target
            dx, dy = tx - cx, ty - cy
            cv2.circle(frame, (tx, ty), 6, (0, 255, 0), -1)
            cv2.line(frame, (cx, cy), (tx, ty), (0, 255, 0), 2)
            cv2.putText(frame, label, (tx - 20, ty - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"dx:{dx:+d}  dy:{dy:+d}",
                        (tx - 20, ty + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)
            cv2.putText(frame, detect_source, (w - 200, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 200), 2)
            status = ("CENTERED"
                      if abs(dx) < CENTER_TOLERANCE
                      and abs(dy) < CENTER_TOLERANCE
                      else "NOT CENTERED")
            color  = (0, 255, 0) if status == "CENTERED" else (0, 0, 255)
        else:
            status, color = "NO TARGET", (0, 0, 255)

        # Crosshair
        cv2.line(frame, (cx-25, cy), (cx+25, cy), (0, 255, 255), 2)
        cv2.line(frame, (cx, cy-25), (cx, cy+25), (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        # HUD
        cv2.putText(frame,
                    f"{mode} (bright:{int(brightness)})",
                    (w - 270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 200, 255) if mode == "NIGHT" else (0, 255, 100), 2)
        cv2.putText(frame, status, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

        cv2.imshow("Integrated Detection", frame)

        with display_lock:
            if shared_balloon_mask is not None:
                cv2.imshow("Balloon Mask",  shared_balloon_mask)
            if shared_balloon_edges is not None:
                cv2.imshow("Balloon Edges", shared_balloon_edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()