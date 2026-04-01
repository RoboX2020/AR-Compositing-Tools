import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import threading
import time

MODEL_PATH = "hand_landmarker.task"

# ─────────────────────────────────────────────
# Threaded Camera Capture
# ─────────────────────────────────────────────
class ThreadedCamera:
    def __init__(self, src=0, name="Camera"):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.name = name
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                if grabbed:
                    self.frame = frame

    def read(self):
        with self.read_lock:
            if not self.grabbed:
                return False, None
            return self.grabbed, self.frame.copy()

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join()
        self.cap.release()

# ─────────────────────────────────────────────
# Geometry Helpers
# ─────────────────────────────────────────────
def order_quad(pts):
    """Order 4 points to top-left, top-right, bottom-right, bottom-left."""
    pts = sorted(pts, key=lambda p: p[1])  # Sort by Y
    top_two = sorted(pts[:2], key=lambda p: p[0])
    bot_two = sorted(pts[2:], key=lambda p: p[0])
    return [top_two[0], top_two[1], bot_two[1], bot_two[0]]

# ─────────────────────────────────────────────
# Main Flow
# ─────────────────────────────────────────────
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found. Run hand_magic.py first.")
        return

    # Hand detector (2 hands)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    hand_options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE)
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    # Note: On macOS, usually 0 is Mac camera, 1 is iPhone (Continuity Camera).
    mac_cam_idx = 0
    phone_cam_idx = 1

    print("=" * 50)
    print("  🎥 AR CAMERA CROP")
    print("=" * 50)
    print(f"Connecting to Mac Camera (idx {mac_cam_idx}) and Phone Camera (idx {phone_cam_idx})...")
    
    # Init cameras
    mac_cam = ThreadedCamera(mac_cam_idx, "Mac").start()
    phone_cam = ThreadedCamera(phone_cam_idx, "Phone").start()

    # Allow cameras to warm up
    time.sleep(1.0)
    
    if not mac_cam.cap.isOpened() or not phone_cam.cap.isOpened():
        print("WARNING: Could not open both cameras. Ensure iPhone is connected and available via Continuity Camera, or try swapping indices.")

    print()
    print("Controls:")
    print("  SPACE - Swap background and overlay cameras")
    print("  R     - Lock/Unlock the region in space")
    print("  Q     - Quit")
    print("=" * 50)

    region_mode = "tracking"
    locked_quad = None

    while True:
        success_bg, bg_frame = mac_cam.read()
        success_fg, fg_frame = phone_cam.read()

        if not success_bg:
            time.sleep(0.01)
            continue
            
        bg_frame = cv2.flip(bg_frame, 1) # Flip Mac frame (mirror)
        # Flip foreground frame as well so it acts like a mirror
        if success_fg:
            fg_frame = cv2.flip(fg_frame, 1)

        h, w, _ = bg_frame.shape

        # Detect hands on the background frame (Mac camera) if tracking
        live_quad = None
        if region_mode == "tracking":
            img_rgb = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            hand_results = hand_detector.detect(mp_image)

            if hand_results.hand_landmarks and len(hand_results.hand_landmarks) >= 2:
                finger_pts = []
                for hand_lm in hand_results.hand_landmarks[:2]:
                    finger_pts.append((int(hand_lm[4].x * w), int(hand_lm[4].y * h)))  # Thumb
                    finger_pts.append((int(hand_lm[8].x * w), int(hand_lm[8].y * h)))  # Index
                
                # Form quad
                live_quad = order_quad(finger_pts)

        active_quad = locked_quad if region_mode == "locked" else live_quad
        
        # Output frame starts as background
        result = bg_frame.copy()

        if active_quad is not None and success_fg:
            # Quad on the background screen
            quad_np = np.array(active_quad, dtype=np.float32)

            # Draw glowing line around the quad
            int_quad = np.int32([active_quad])
            border_color = (0, 255, 0) if region_mode == "locked" else (0, 255, 255)
            cv2.polylines(result, int_quad, True, border_color, 2, cv2.LINE_AA)
            for pt in active_quad:
                cv2.circle(result, pt, 8, border_color, -1, cv2.LINE_AA)

            # Get dimensions of overlay frame
            fh, fw, _ = fg_frame.shape
            
            # The 4 corners of the full foreground image
            fg_corners = np.array([
                [0, 0],
                [fw, 0],
                [fw, fh],
                [0, fh]
            ], dtype=np.float32)

            # Calculate perspective transform to map fg bounds -> finger quad
            matrix = cv2.getPerspectiveTransform(fg_corners, quad_np)
            
            # Warp the foreground image
            warped_fg = cv2.warpPerspective(fg_frame, matrix, (w, h))

            # Cut out the region in the background to place the warped foreground
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(active_quad), 255)
            
            # Composite
            mask_inv = cv2.bitwise_not(mask)
            bg_masked = cv2.bitwise_and(result, result, mask=mask_inv)
            result = cv2.add(bg_masked, warped_fg)

        # HUD
        cv2.rectangle(result, (0, 0), (w, 40), (10, 10, 15), -1)
        cv2.putText(result, "AR CAMERA CROP", (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2, cv2.LINE_AA)

        if region_mode == "locked":
            status = "TRACKING: Region LOCKED (Press R to unlock)"
            color = (0, 255, 0)
        elif active_quad is not None:
            status = "TRACKING: 2 Hands Found (Press R to lock)"
            color = (0, 255, 255)
        else:
            status = "TRACKING: Show 2 hands (Thumb + Index)"
            color = (0, 100, 255)
            
        cv2.putText(result, status, (350, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        cv2.putText(result, "SPACE: Swap | R: Lock | Q: Quit", (950, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("AR Camera Crop", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('r') or key == ord('R'):
            if region_mode == "tracking":
                if live_quad is not None:
                    locked_quad = live_quad
                    region_mode = "locked"
                    print("Region LOCKED in space. Press R to unlock.")
                else:
                    print("Cannot lock: hands not detected.")
            elif region_mode == "locked":
                region_mode = "tracking"
                locked_quad = None
                print("Region UNLOCKED. Tracking hands again.")
        elif key == ord(' '):
            # Swap cameras
            print("Swapping cameras...")
            tmp = mac_cam
            mac_cam = phone_cam
            phone_cam = tmp

    mac_cam.stop()
    phone_cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
