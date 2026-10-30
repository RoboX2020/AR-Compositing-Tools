# ====================================================================
# live_filters.py - webcam fun-mirror filters (anime, x-ray, thermal,
# neon edges and friends): each filter is a small OpenCV recipe applied
# to every frame.
# ====================================================================
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
import math

MODEL_PATH = "hand_landmarker.task"

# ─────────────────────────────────────────────
# Filter Functions
# ─────────────────────────────────────────────

def anime_filter(frame):
    """Convert to anime/cartoon style: smooth colors + bold edges."""
    # Bilateral filter for smooth flat colors
    smooth = frame.copy()
    for _ in range(3):
        smooth = cv2.bilateralFilter(smooth, 9, 75, 75)

    # Fast color quantization via bit reduction (no kmeans!)
    div = 32
    quantized = (smooth // div) * div + div // 2

    # Boost saturation for vivid anime colors
    hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255).astype(np.uint8)
    vivid = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Bold black edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 9, 5)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    result = cv2.bitwise_and(vivid, edges_bgr)
    return result


def xray_filter(frame):
    """Simulate X-ray / radiograph look."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Invert
    inverted = cv2.bitwise_not(gray)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(inverted)

    # Edge detection for "bone structure"
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, None, iterations=1)

    # Combine: bright edges on inverted image
    result = cv2.addWeighted(enhanced, 0.8, edges, 0.5, 0)

    # Add cool blue tint
    colored = np.zeros_like(frame)
    colored[:, :, 0] = np.clip(result * 1.0, 0, 255).astype(np.uint8)   # Blue
    colored[:, :, 1] = np.clip(result * 0.8, 0, 255).astype(np.uint8)   # Green
    colored[:, :, 2] = np.clip(result * 0.3, 0, 255).astype(np.uint8)   # Less red

    # Add glow to bright areas (simulating bone density)
    bright = cv2.GaussianBlur(result, (15, 15), 0)
    glow = np.zeros_like(frame)
    glow[:, :, 0] = np.clip(bright * 0.3, 0, 255).astype(np.uint8)
    glow[:, :, 1] = np.clip(bright * 0.6, 0, 255).astype(np.uint8)
    glow[:, :, 2] = np.clip(bright * 0.1, 0, 255).astype(np.uint8)

    colored = cv2.add(colored, glow)
    return colored


def thermal_filter(frame):
    """Thermal / infrared camera look."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE for detail
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Blur slightly
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply thermal colormap
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return thermal


def neon_edges_filter(frame):
    """Glowing neon edge detection effect."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Multi-scale edge detection
    edges1 = cv2.Canny(gray, 50, 150)
    edges2 = cv2.Canny(gray, 20, 80)

    # Dilate for thickness
    edges1 = cv2.dilate(edges1, None, iterations=1)
    edges2 = cv2.dilate(edges2, None, iterations=1)

    # Create neon colored edges
    result = np.zeros_like(frame)
    # Cyan edges
    result[:, :, 0] = np.clip(edges1.astype(np.int16) * 1.0, 0, 255).astype(np.uint8)
    result[:, :, 1] = np.clip(edges1.astype(np.int16) * 0.9, 0, 255).astype(np.uint8)
    # Pink secondary edges
    result[:, :, 2] = np.clip(result[:, :, 2].astype(np.int16) + edges2.astype(np.int16) * 0.8, 0, 255).astype(np.uint8)
    result[:, :, 0] = np.clip(result[:, :, 0].astype(np.int16) + edges2.astype(np.int16) * 0.3, 0, 255).astype(np.uint8)

    # Glow effect via blur
    glow = cv2.GaussianBlur(result, (9, 9), 0)
    result = cv2.add(result, glow)

    # Add faint dark background
    dark_bg = cv2.convertScaleAbs(frame, alpha=0.1, beta=0)
    result = cv2.add(result, dark_bg)

    return result


def pencil_sketch_filter(frame):
    """Pencil sketch / drawing effect."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, cv2.bitwise_not(blur), scale=256)

    # Add paper texture tint
    colored = np.zeros_like(frame)
    colored[:, :, 0] = np.clip(sketch * 0.85, 0, 255).astype(np.uint8)  # Slightly warm
    colored[:, :, 1] = np.clip(sketch * 0.9, 0, 255).astype(np.uint8)
    colored[:, :, 2] = np.clip(sketch * 0.95, 0, 255).astype(np.uint8)
    return colored


def oil_painting_filter(frame):
    """Oil painting effect."""
    # Heavy bilateral filtering for painterly look
    result = frame.copy()
    for _ in range(4):
        result = cv2.bilateralFilter(result, 9, 75, 75)

    # Boost saturation
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Add subtle edge darkening
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, 5)
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(result, edges_3ch)
    return result


def pixel_art_filter(frame):
    """Retro pixel art / 8-bit effect."""
    pixel_size = 8
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)

    # Fast color quantization via bit reduction
    div = 48
    quantized = (small // div) * div + div // 2

    # Upsample with nearest neighbor (crisp pixels)
    result = cv2.resize(quantized, (w, h), interpolation=cv2.INTER_NEAREST)
    return result


def glitch_filter(frame):
    """RGB channel shift glitch effect."""
    h, w = frame.shape[:2]
    result = frame.copy()

    # Random horizontal shift per channel
    shift_r = np.random.randint(-15, 15)
    shift_b = np.random.randint(-15, 15)

    result[:, :, 2] = np.roll(frame[:, :, 2], shift_r, axis=1)  # Red
    result[:, :, 0] = np.roll(frame[:, :, 0], shift_b, axis=1)  # Blue

    # Random scanline corruption
    for _ in range(5):
        y = np.random.randint(0, h)
        thickness = np.random.randint(1, 4)
        shift = np.random.randint(-30, 30)
        result[y:y+thickness] = np.roll(result[y:y+thickness], shift, axis=1)

    # Random color block
    if np.random.random() < 0.3:
        bx = np.random.randint(0, w - 50)
        by = np.random.randint(0, h - 20)
        bw = np.random.randint(30, 100)
        bh = np.random.randint(5, 20)
        result[by:by+bh, bx:bx+bw] = result[by:by+bh, bx:bx+bw][:, :, ::-1]  # Swap channels

    return result


# ─────────────────────────────────────────────
# Filter Registry
# ─────────────────────────────────────────────
FILTERS = [
    ("ANIME / CARTOON", anime_filter),
    ("X-RAY", xray_filter),
    ("THERMAL VISION", thermal_filter),
    ("NEON EDGES", neon_edges_filter),
    ("PENCIL SKETCH", pencil_sketch_filter),
    ("OIL PAINTING", oil_painting_filter),
    ("PIXEL ART / 8-BIT", pixel_art_filter),
    ("GLITCH", glitch_filter),
]


# ─────────────────────────────────────────────
# Order quad corners: TL, TR, BR, BL
# ─────────────────────────────────────────────
def order_quad(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = sorted(pts, key=lambda p: p[1])  # Sort by Y
    top_two = sorted(pts[:2], key=lambda p: p[0])
    bot_two = sorted(pts[2:], key=lambda p: p[0])
    return [top_two[0], top_two[1], bot_two[1], bot_two[0]]


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found! Run hand_magic.py first to download.")
        return

    # Hand detector
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    hand_options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE)
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    cam_idx = 0
    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_filter = 0

    # Region state: "off" -> "selecting" -> "locked" -> "off"
    region_mode = "off"
    locked_quad = None  # 4 points when locked

    print("=" * 50)
    print("  🎬 LIVE VIDEO FILTERS + HAND CROP")
    print("=" * 50)
    print()
    for i, (name, _) in enumerate(FILTERS):
        print(f"  {i+1}. {name}")
    print()
    print("Controls:")
    print("  1-8 / Arrows   - Switch filter")
    print("  R              - Region crop (toggle)")
    print("    → Selecting: show BOTH hands, thumb+index = 4 corners")
    print("    → Press R to lock region")
    print("    → Press R again to go full-screen")
    print("  N              - Switch camera")
    print("  Q              - Quit")
    print("=" * 50)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Apply current filter to FULL frame
        filter_name, filter_fn = FILTERS[current_filter]
        try:
            filtered = filter_fn(frame)
        except Exception:
            filtered = frame.copy()

        # ── Hand detection for region selection ──
        live_quad = None
        if region_mode == "selecting":
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            hand_results = hand_detector.detect(mp_image)

            if hand_results.hand_landmarks and len(hand_results.hand_landmarks) >= 2:
                # Use thumb(4) + index(8) from each hand = 4 corners
                finger_pts = []
                for hand_lm in hand_results.hand_landmarks[:2]:
                    finger_pts.append((int(hand_lm[4].x * w), int(hand_lm[4].y * h)))  # Thumb
                    finger_pts.append((int(hand_lm[8].x * w), int(hand_lm[8].y * h)))  # Index
                live_quad = order_quad(finger_pts)

        # ── Compose the result ──
        active_quad = locked_quad if region_mode == "locked" else live_quad

        if active_quad is not None:
            # Build mask from the quad
            quad_np = np.array(active_quad, dtype=np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, quad_np, 255)
            mask_3ch = cv2.merge([mask, mask, mask])
            inv_mask_3ch = cv2.bitwise_not(mask_3ch)

            # Dark normal feed outside, filtered inside
            dark_bg = cv2.convertScaleAbs(frame, alpha=0.35, beta=5)
            result = cv2.bitwise_and(filtered, mask_3ch) + cv2.bitwise_and(dark_bg, inv_mask_3ch)

            # Draw the quad border
            border_color = (0, 255, 0) if region_mode == "locked" else (0, 255, 255)
            cv2.polylines(result, [quad_np], True, border_color, 3, cv2.LINE_AA)

            # Label corners
            labels = ["1", "2", "3", "4"]
            for i, pt in enumerate(active_quad):
                cv2.circle(result, pt, 8, border_color, -1, cv2.LINE_AA)
                cv2.circle(result, pt, 12, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(result, labels[i], (pt[0] + 15, pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2, cv2.LINE_AA)
        else:
            result = filtered

        # HUD
        cv2.rectangle(result, (0, 0), (w, 50), (10, 10, 15), -1)
        cv2.putText(result, f"FILTER: {filter_name}", (30, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)

        region_label = ""
        if region_mode == "selecting":
            region_label = " | REGION: Show 4 fingers, press R to lock"
        elif region_mode == "locked":
            region_label = " | REGION: LOCKED (R to unlock)"

        cv2.putText(result, f"[{current_filter+1}/{len(FILTERS)}] 1-8/Arrows | R: Crop | N: Cam{region_label}",
                    (430, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        cv2.imshow("Live Video Filters", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == 83 or key == ord('.'):
            current_filter = (current_filter + 1) % len(FILTERS)
            print(f"Filter: {FILTERS[current_filter][0]}")
        elif key == 81 or key == ord(','):
            current_filter = (current_filter - 1) % len(FILTERS)
            print(f"Filter: {FILTERS[current_filter][0]}")
        elif key == ord('r') or key == ord('R'):
            if region_mode == "off":
                region_mode = "selecting"
                locked_quad = None
                print("Region select: show 4 fingers, then press R to lock")
            elif region_mode == "selecting":
                if live_quad is not None:
                    locked_quad = live_quad
                    region_mode = "locked"
                    print("Region LOCKED! Press R to go back to full-screen")
                else:
                    print("No hand detected — show your hand with 4 fingers spread")
            elif region_mode == "locked":
                region_mode = "off"
                locked_quad = None
                print("Region cleared — full-screen mode")
        elif key == ord('n') or key == ord('N'):
            cap.release()
            cam_idx = (cam_idx + 1) % 3
            cap = cv2.VideoCapture(cam_idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print(f"Camera: {cam_idx}")
        elif ord('1') <= key <= ord('8'):
            idx = key - ord('1')
            if idx < len(FILTERS):
                current_filter = idx
                print(f"Filter: {FILTERS[current_filter][0]}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
