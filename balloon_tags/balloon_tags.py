import cv2
import cv2.aruco as aruco
from pupil_apriltags import Detector
import numpy as np
import threading
import queue
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point

# ─────────────────────────────────────────────
#  CAMERA — GStreamer with fallback
# ─────────────────────────────────────────────
def open_camera():
    gst_pipeline = (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw, width=1280, height=720, framerate=30/1 ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=1"
    )
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("[INFO] Camera opened with GStreamer")
        return cap

    print("[WARN] GStreamer failed — falling back to normal capture")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("[INFO] Camera opened with default backend")
        return cap

    print("[ERROR] Cannot open camera at all")
    exit(1)

cap = open_camera()
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

# ─────────────────────────────────────────────
#  QR DETECTOR
# ─────────────────────────────────────────────
try:
    qr_detector = cv2.QRCodeDetectorAruco()
    print("[INFO] QRCodeDetectorAruco loaded")
except AttributeError:
    qr_detector = cv2.QRCodeDetector()
    print("[INFO] QRCodeDetector (fallback) loaded")

# ─────────────────────────────────────────────
#  ARUCO
# ─────────────────────────────────────────────
ARUCO_DICTS = [
    aruco.DICT_4X4_1000,
    aruco.DICT_5X5_1000,
    aruco.DICT_6X6_1000,
    aruco.DICT_7X7_1000,
]
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
BALLOON_AREA_THRESHOLD = 3000
BALLOON_DIAMETER_M     = 0.25
BALLOON_RESULT_TTL     = 0.15
BALLOON_SCALE          = 0.5

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
MAX_LOST            = 20
NIGHT_THRESHOLD     = 80
MIN_MARKER_AREA     = 2000
MIN_QR_AREA         = 1500
QR_RESULT_TTL       = 0.4
SPATIAL_MERGE_DIST  = 80
ALIGNMENT_THRESHOLD = 30

# ─────────────────────────────────────────────
#  SHARED DISPLAY FRAMES (thread → main)
# ─────────────────────────────────────────────
display_lock         = threading.Lock()
shared_balloon_mask  = None
shared_balloon_edges = None

# ─────────────────────────────────────────────
#  QR BACKGROUND THREAD
# ─────────────────────────────────────────────
qr_input_queue  = queue.Queue(maxsize=1)
qr_result_queue = queue.Queue(maxsize=1)

def preprocess_for_qr(gray):
    sharp = cv2.filter2D(gray, -1, SHARPEN_KERNEL)
    return cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

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
                pts  = pts_list[best_i].astype(int)
                result = (int(pts[:, 0].mean()), int(pts[:, 1].mean()),
                          data_list[best_i], pts)
    except Exception:
        try:
            data, pts, _ = qr_detector.detectAndDecode(img)
            if data and pts is not None:
                pts = pts[0].astype(int)
                if cv2.contourArea(pts.astype(np.float32)) >= MIN_QR_AREA:
                    result = (int(pts[:, 0].mean()), int(pts[:, 1].mean()), data, pts)
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
            result = try_detect_qr(cv2.filter2D(gray, -1, SHARPEN_KERNEL), cx, cy)
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
        if not qr_result_queue.full():
            qr_result_queue.put((result, time.monotonic()))
        else:
            try:
                qr_result_queue.get_nowait()
            except queue.Empty:
                pass
            qr_result_queue.put((result, time.monotonic()))

threading.Thread(target=qr_worker, daemon=True).start()

# ─────────────────────────────────────────────
#  BALLOON BACKGROUND THREAD
# ─────────────────────────────────────────────
balloon_input_queue  = queue.Queue(maxsize=1)
balloon_result_queue = queue.Queue(maxsize=1)

def is_balloon_contour(cnt, min_circularity=0.65):
    area  = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    if perim == 0:
        return False
    return (4 * np.pi * area / (perim ** 2)) > min_circularity

def white_balance(img):
    result              = img.astype(np.float32)
    avg_b, avg_g, avg_r = (np.mean(result[:, :, c]) for c in range(3))
    avg_gray            = (avg_b + avg_g + avg_r) / 3
    result[:, :, 0]    *= avg_gray / (avg_b + 1e-6)
    result[:, :, 1]    *= avg_gray / (avg_g + 1e-6)
    result[:, :, 2]    *= avg_gray / (avg_r + 1e-6)
    return np.clip(result, 0, 255).astype(np.uint8)

def balloon_worker():
    global shared_balloon_mask, shared_balloon_edges
    local_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    while True:
        try:
            frame_small, scale, full_w, full_h = balloon_input_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        wb       = white_balance(frame_small)
        lab      = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
        l, a, b  = cv2.split(lab)
        l        = local_clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        blurred  = cv2.GaussianBlur(enhanced, (7, 7), 0)
        hsv      = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray_s   = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        edges_s  = cv2.Canny(gray_s, 50, 150)

        best_contour   = None
        detected_color = None
        max_area       = 0
        best_mask_s    = None

        for color_name, ranges in COLOR_RANGES.items():
            mask = None
            for lower, upper in ranges:
                temp = cv2.inRange(hsv, lower, upper)
                mask = temp if mask is None else cv2.bitwise_or(mask, temp)

            mask = cv2.medianBlur(mask, 5)
            mask = cv2.erode(mask,  None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

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

        mask_full = cv2.resize(
            best_mask_s if best_mask_s is not None
            else np.zeros((frame_small.shape[0], frame_small.shape[1]), dtype=np.uint8),
            (full_w, full_h), interpolation=cv2.INTER_NEAREST)

        edges_full = cv2.resize(edges_s, (full_w, full_h),
                                interpolation=cv2.INTER_NEAREST)

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

        if not balloon_result_queue.full():
            balloon_result_queue.put((result, time.monotonic()))
        else:
            try:
                balloon_result_queue.get_nowait()
            except queue.Empty:
                pass
            balloon_result_queue.put((result, time.monotonic()))

threading.Thread(target=balloon_worker, daemon=True).start()

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def is_valid_marker(pts):
    area = cv2.contourArea(pts.astype(np.float32))
    if area < MIN_MARKER_AREA:
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

# ─────────────────────────────────────────────
#  ROS2 NODE
# ─────────────────────────────────────────────
class DetectionNode(Node):
    def __init__(self):
        super().__init__('balloon_tags_node')

        # Publishers
        self.pub_label  = self.create_publisher(String, '/detection/label',  10)
        self.pub_offset = self.create_publisher(Point,  '/detection/offset', 10)
        self.pub_source = self.create_publisher(String, '/detection/source', 10)

        # State
        self.last_target         = None
        self.lost_frames         = 0
        self.last_qr_result      = None
        self.last_qr_ts          = 0.0
        self.last_balloon_result = None
        self.last_balloon_ts     = 0.0

        # Timer ~30Hz
        self.timer = self.create_timer(0.033, self.loop)
        self.get_logger().info("DetectionNode started — press Q in window to quit")

    def loop(self):
        global shared_balloon_mask, shared_balloon_edges

        ret, frame = cap.read()
        if not ret:
            self.get_logger().warn("Cannot read frame")
            return

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

        detected      = None
        detect_source = ""
        marker_found  = False

        # ════════════════════════════════════════════════════════════
        #  1. ARUCO
        # ════════════════════════════════════════════════════════════
        all_detections = []
        for det_i, aruco_detector in enumerate(aruco_detectors):
            corners, ids, _ = aruco_detector.detectMarkers(enhanced)
            if ids is None:
                corners, ids, _ = aruco_detector.detectMarkers(gray)
            if ids is None or len(ids) == 0:
                continue
            for i in range(len(ids)):
                c = corners[i][0]
                if not is_valid_marker(c):
                    continue
                area = cv2.contourArea(c)
                tx   = int(c[:, 0].mean())
                ty   = int(c[:, 1].mean())
                all_detections.append((
                    ARUCO_DICT_PRIORITY[det_i], area, tx, ty,
                    int(ids[i][0]), ARUCO_DICT_NAMES[det_i],
                    corners[i], ids[i]
                ))

        if all_detections:
            used   = [False] * len(all_detections)
            groups = []
            for i in range(len(all_detections)):
                if used[i]:
                    continue
                group   = [i]
                used[i] = True
                for j in range(i + 1, len(all_detections)):
                    if used[j]:
                        continue
                    dist = ((all_detections[i][2] - all_detections[j][2]) ** 2 +
                            (all_detections[i][3] - all_detections[j][3]) ** 2) ** 0.5
                    if dist < SPATIAL_MERGE_DIST:
                        group.append(j)
                        used[j] = True
                groups.append(group)

            best_detection, best_score = None, (-1, -1)
            for group in groups:
                for idx in group:
                    pri, area = all_detections[idx][0], all_detections[idx][1]
                    if (pri, area) > best_score:
                        best_score     = (pri, area)
                        best_detection = all_detections[idx]

            if best_detection:
                _, area, tx, ty, aruco_id, dict_name, corner, aid = best_detection
                detected      = (tx, ty, f"ArUco {dict_name}:{aruco_id}")
                detect_source = f"ARUCO-{dict_name}"
                marker_found  = True
                aruco.drawDetectedMarkers(frame, [corner], np.array([[aruco_id]]))
                self.get_logger().info(f"[ArUco-{dict_name}] ID:{aruco_id}  X:{tx}  Y:{ty}")

        # ════════════════════════════════════════════════════════════
        #  2. APRILTAG
        # ════════════════════════════════════════════════════════════
        if not marker_found:
            tags = apriltag_detector.detect(enhanced)
            for tag in tags:
                tx, ty        = int(tag.center[0]), int(tag.center[1])
                detected      = (tx, ty, f"April {tag.tag_id}")
                detect_source = "APRIL"
                marker_found  = True
                self.get_logger().info(f"[AprilTag] ID:{tag.tag_id}  X:{tx}  Y:{ty}")
                break

        # ════════════════════════════════════════════════════════════
        #  3. QR
        # ════════════════════════════════════════════════════════════
        if not marker_found:
            if qr_input_queue.empty():
                try:
                    qr_input_queue.put_nowait((gray.copy(), cx, cy))
                except queue.Full:
                    pass
            try:
                qr_res, qr_ts       = qr_result_queue.get_nowait()
                self.last_qr_result = qr_res
                self.last_qr_ts     = qr_ts
            except queue.Empty:
                pass

            if self.last_qr_result is not None and (now - self.last_qr_ts) < QR_RESULT_TTL:
                tx, ty, data, pts = self.last_qr_result
                detected      = (tx, ty, f"QR:{data}")
                detect_source = "QR"
                marker_found  = True
                for i in range(len(pts)):
                    cv2.line(frame, tuple(pts[i]),
                             tuple(pts[(i + 1) % len(pts)]), (0, 255, 0), 2)
                cv2.putText(frame, f"{data[:30]}", (tx - 20, ty + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                self.get_logger().info(f"[QR] Data:{data}  X:{tx}  Y:{ty}")
            else:
                if (now - self.last_qr_ts) >= QR_RESULT_TTL:
                    self.last_qr_result = None

        # ════════════════════════════════════════════════════════════
        #  4. BALLOON
        # ════════════════════════════════════════════════════════════
        if balloon_input_queue.empty():
            try:
                small = cv2.resize(frame, (0, 0), fx=BALLOON_SCALE, fy=BALLOON_SCALE)
                balloon_input_queue.put_nowait((small, BALLOON_SCALE, w, h))
            except queue.Full:
                pass

        try:
            b_res, b_ts              = balloon_result_queue.get_nowait()
            self.last_balloon_result = b_res
            self.last_balloon_ts     = b_ts
        except queue.Empty:
            pass

        if not marker_found:
            if self.last_balloon_result is not None and (now - self.last_balloon_ts) < BALLOON_RESULT_TTL:
                b_center, b_radius, b_color, b_contour = self.last_balloon_result
                bx, by = b_center
                dx_px  = bx - cx
                dy_px  = cy - by

                meters_per_pixel = BALLOON_DIAMETER_M / (2 * b_radius + 1e-6)
                dx_m   = dx_px * meters_per_pixel
                dy_m   = dy_px * meters_per_pixel
                dist_m = np.sqrt(dx_m ** 2 + dy_m ** 2)
                aligned = (abs(dx_px) <= ALIGNMENT_THRESHOLD and
                           abs(dy_px) <= ALIGNMENT_THRESHOLD)

                detected      = (bx, by, f"Balloon:{b_color}")
                detect_source = "BALLOON"
                self.get_logger().info(f"[Balloon] {b_color}  X:{bx}  Y:{by}")

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
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
                else:
                    moves = []
                    if abs(dx_px) > ALIGNMENT_THRESHOLD:
                        moves.append(f"{'RIGHT' if dx_px > 0 else 'LEFT'} {abs(dx_m):.2f}m")
                    if abs(dy_px) > ALIGNMENT_THRESHOLD:
                        moves.append(f"{'UP' if dy_px > 0 else 'DOWN'} {abs(dy_m):.2f}m")
                    cv2.putText(frame, "MOVE: " + " | ".join(moves),
                                (20, 155), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 255), 2)
            else:
                if (now - self.last_balloon_ts) >= BALLOON_RESULT_TTL:
                    self.last_balloon_result = None

        # ════════════════════════════════════════════════════════════
        #  STABILITY FILTER
        # ════════════════════════════════════════════════════════════
        if detected is not None:
            self.last_target = detected
            self.lost_frames = 0
        else:
            self.lost_frames += 1
            if self.lost_frames >= MAX_LOST:
                self.last_target         = None
                self.last_qr_result      = None
                self.last_balloon_result = None

        # ════════════════════════════════════════════════════════════
        #  PUBLISH TO ROS2
        # ════════════════════════════════════════════════════════════
        if self.last_target is not None and self.lost_frames < MAX_LOST:
            tx, ty, label = self.last_target
            dx, dy = tx - cx, ty - cy

            msg_label       = String()
            msg_label.data  = label
            self.pub_label.publish(msg_label)

            msg_offset      = Point()
            msg_offset.x    = float(dx)
            msg_offset.y    = float(dy)
            msg_offset.z    = 0.0
            self.pub_offset.publish(msg_offset)

            msg_source      = String()
            msg_source.data = detect_source
            self.pub_source.publish(msg_source)

            cv2.circle(frame, (tx, ty), 6, (0, 255, 0), -1)
            cv2.line(frame, (cx, cy), (tx, ty), (0, 255, 0), 2)
            cv2.putText(frame, label, (tx - 20, ty - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"dx:{dx:+d}  dy:{dy:+d}", (tx - 20, ty + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)
            cv2.putText(frame, detect_source, (w - 200, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 200), 2)
            status = "CENTERED" if abs(dx) < CENTER_TOLERANCE and abs(dy) < CENTER_TOLERANCE else "NOT CENTERED"
            color  = (0, 255, 0) if status == "CENTERED" else (0, 0, 255)
        else:
            status, color = "NO TARGET", (0, 0, 255)

        # Crosshair
        cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (0, 255, 255), 2)
        cv2.line(frame, (cx, cy - 25), (cx, cy + 25), (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        # HUD
        cv2.putText(frame, f"{mode} (bright:{int(brightness)})", (w - 270, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 200, 255) if mode == "NIGHT" else (0, 255, 100), 2)
        cv2.putText(frame, status, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

        # Show windows
        cv2.imshow("Integrated Detection", frame)

        with display_lock:
            if shared_balloon_mask is not None:
                cv2.imshow("Balloon Mask", shared_balloon_mask)
            if shared_balloon_edges is not None:
                cv2.imshow("Edges", shared_balloon_edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cap.release()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()