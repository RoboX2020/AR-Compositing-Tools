# ====================================================================
# ar_compositing.py - green screen without a green screen: an AI model
# (MediaPipe selfie segmenter) cuts you out of your background in real
# time so you can be placed over anything.
# ====================================================================
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import threading
import time
import urllib.request

MODEL_PATH = "selfie_segmenter.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"

# ─────────────────────────────────────────────
# Threaded Camera Capture
# ─────────────────────────────────────────────
# # Reads webcam frames on a background thread so the main loop never waits
# # on the camera - the display stays smooth even if a frame is slow.
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
# Main
# ─────────────────────────────────────────────
def main():
    # Download the segmentation model if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading {MODEL_PATH} (this only happens once)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")

    # Initialize the Selfie Segmenter
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=False,
        output_confidence_masks=True)
    segmenter = vision.ImageSegmenter.create_from_options(options)

    # Mac and iPhone cameras
    mac_cam_idx = 0
    phone_cam_idx = 1

    print("=" * 50)
    print("  🎬 AR LIVE GREENSCREEN")
    print("=" * 50)
    print(f"Connecting to BG Camera (idx {mac_cam_idx}) and FG Camera (idx {phone_cam_idx})...")
    
    mac_cam = ThreadedCamera(mac_cam_idx, "Mac BG").start()
    phone_cam = ThreadedCamera(phone_cam_idx, "Phone FG").start()

    time.sleep(1.0)
    
    if not mac_cam.cap.isOpened() or not phone_cam.cap.isOpened():
        print("WARNING: Could not open both cameras. Try swapping indices if it crashes.")

    # Overlay scale and position
    scale_percent = 1.0
    offset_x, offset_y = 0, 0
    bg_is_mac = True

    print()
    print("Controls:")
    print("  SPACE   - Swap BG and Segemented cameras")
    print("  W/A/S/D - Move the cut-out person")
    print("  +/-     - Scale the person up/down")
    print("  Q       - Quit")
    print("=" * 50)

    while True:
        success_mac, mac_frame = mac_cam.read()
        success_ph, ph_frame = phone_cam.read()

        if not success_mac or not success_ph:
            time.sleep(0.01)
            continue
            
        mac_frame = cv2.flip(mac_frame, 1)
        ph_frame = cv2.flip(ph_frame, 1)

        # Decide which is background and which is foreground (person)
        bg_frame = mac_frame if bg_is_mac else ph_frame
        fg_frame = ph_frame if bg_is_mac else mac_frame

        h, w, _ = bg_frame.shape
        fh, fw, _ = fg_frame.shape

        # Resize FG frame just in case their resolutions differ fundamentally
        if (fh, fw) != (h, w):
            fg_frame = cv2.resize(fg_frame, (w, h))
            fh, fw = h, w

        # 1. Run MediaPipe Segmentation on Foreground
        img_rgb = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        seg_result = segmenter.segment(mp_image)
        
        # 2. Get Confidence Mask (Alpha channel)
        # Often class 1 is the person and class 0 is background in this model
        if len(seg_result.confidence_masks) > 1:
            person_mask = seg_result.confidence_masks[1].numpy_view()
        else:
            person_mask = seg_result.confidence_masks[0].numpy_view()
            
        # Blur the mask slightly so the edges look nice and smooth
        alpha = cv2.GaussianBlur(person_mask, (7, 7), 0)

        # 3. Handle Scaling & Repositioning
        new_fw, new_fh = int(fw * scale_percent), int(fh * scale_percent)
        fg_scaled = cv2.resize(fg_frame, (new_fw, new_fh))
        alpha_scaled = cv2.resize(alpha, (new_fw, new_fh))
        
        # We need a 3-channel alpha for cv2 mathematics
        alpha_3 = np.stack([alpha_scaled]*3, axis=-1)

        result = bg_frame.copy()

        # Calculate bounding boxes
        x1 = max(0, offset_x)
        y1 = max(0, offset_y)
        x2 = min(w, offset_x + new_fw)
        y2 = min(h, offset_y + new_fh)

        fg_x1 = max(0, -offset_x)
        fg_y1 = max(0, -offset_y)
        fg_x2 = fg_x1 + (x2 - x1)
        fg_y2 = fg_y1 + (y2 - y1)

        if x1 < x2 and y1 < y2:
            bg_roi = result[y1:y2, x1:x2]
            fg_roi = fg_scaled[fg_y1:fg_y2, fg_x1:fg_x2]
            a_roi = alpha_3[fg_y1:fg_y2, fg_x1:fg_x2]

            # 4. Composite (blend foreground and background)
            blended = fg_roi * a_roi + bg_roi * (1.0 - a_roi)
            result[y1:y2, x1:x2] = blended.astype(np.uint8)

        # HUD Layer
        cv2.rectangle(result, (0, 0), (w, 40), (10, 10, 15), -1)
        cv2.putText(result, "AR LIVE GREENSCREEN", (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)

        cam_text = "Mac=BG, Phone=Person" if bg_is_mac else "Phone=BG, Mac=Person"
        cv2.putText(result, f"Cameras: {cam_text}", (350, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 2, cv2.LINE_AA)

        cv2.putText(result, "SPACE: Swap | WASD: Move | +/-: Scale | Q: Quit", (810, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("AR Compositing", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord(' '):
            bg_is_mac = not bg_is_mac
            print(f"Swapped! Background is now {'Mac' if bg_is_mac else 'Phone'}.")
        elif key == ord('w'): offset_y -= 25
        elif key == ord('s'): offset_y += 25
        elif key == ord('a'): offset_x -= 25
        elif key == ord('d'): offset_x += 25
        elif key == ord('=') or key == ord('+'):
            scale_percent = min(3.0, scale_percent + 0.1)
        elif key == ord('-') or key == ord('_'):
            scale_percent = max(0.2, scale_percent - 0.1)

    mac_cam.stop()
    phone_cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
