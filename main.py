import cv2
import numpy as np
import math

# Define the common tent colors in HSV format
# Format: 'ColorName': [(Lower_HSV), (Upper_HSV)]
# Note: Red wraps around 0° in HSV so it needs two ranges combined with bitwise_or
TENT_COLORS = {
    'Blue':       (np.array([100, 150,  50]), np.array([130, 255, 255])),
    'Neon Green': (np.array([ 35, 100, 100]), np.array([ 85, 255, 255])),
    'White':      (np.array([  0,   0, 200]), np.array([180,  30, 255])),
    'Orange':     (np.array([  5, 150, 150]), np.array([ 15, 255, 255])),
    'Forest Green': (np.array([40, 40, 30]), np.array([80, 180, 150])),
    'Red':        (np.array([  0, 150, 150]), np.array([ 10, 255, 255])),
    'Red2':       (np.array([170, 150, 150]), np.array([180, 255, 255])),
}

MORPH_KERNEL = np.ones((5, 5), np.uint8)

def get_mask(hsv: np.ndarray, color_name: str) -> np.ndarray:
    """
    Build a binary color mask for the given color.
    Red is special: its hue wraps around 0°/180° so we OR two sub-ranges together.
    """
    lower, upper = TENT_COLORS[color_name]
    mask = cv2.inRange(hsv, lower, upper)

    if color_name == 'Red':
        lower2, upper2 = TENT_COLORS['Red2']
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower2, upper2))

    return mask


def detect_tent(frame: np.ndarray,
                min_area: int = 500,
                max_area: int = 50000) -> tuple[np.ndarray, list[dict]]:
    """
    Detect tents in a single frame.

    Args:
        frame:    BGR image (numpy array).
        min_area: Minimum contour area in pixels. Tune for drone altitude.
        max_area: Maximum contour area in pixels. Tune for drone altitude.

    Returns:
        frame:      The original frame with detection overlays drawn on it.
        detections: List of dicts, each with keys:
                    'color', 'x', 'y', 'w', 'h', 'area', 'circularity'
    """
    # Blur first — reduces compression noise / colour speckle before masking
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (7, 7), 0)

    detections = []

    colors_to_check = [c for c in TENT_COLORS if c != 'Red2']

    for color_name in colors_to_check:
        mask = get_mask(hsv, color_name)

        # OPEN removes small noise specks; CLOSE fills holes from shadows/wrinkles
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  MORPH_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if not (min_area < area < max_area):
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            # Isoperimetric circularity: 1.0 = perfect circle, ~0.785 = square
            circularity = (4 * math.pi * area) / (perimeter ** 2)

            if not (0.5 < circularity < 1.0):
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Tents viewed from above aren't extremely elongated
            aspect_ratio = w / h
            if not (0.5 < aspect_ratio < 2.0):
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{color_name} Tent ({circularity:.2f})"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            detections.append({
                'color':       color_name,
                'x': x, 'y': y, 'w': w, 'h': h,
                'area':        area,
                'circularity': circularity,
            })

    return frame, detections


if __name__ == "__main__":
    import sys

    # Accept an optional CLI argument: python tent_detector.py path/to/image.jpg
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_media/test4.jpg"

    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error: Could not load image at '{image_path}'. Check the file path!")
        sys.exit(1)

    height, width = frame.shape[:2]
    if width > 1280:
        scale = 1280 / width
        frame = cv2.resize(frame, (1280, int(height * scale)))

    processed_frame, detections = detect_tent(frame)

    if detections:
        print(f"Found {len(detections)} tent(s):")
        for d in detections:
            print(f"  {d['color']:12s}  area={d['area']:.0f}px  "
                  f"circularity={d['circularity']:.2f}  "
                  f"bbox=({d['x']},{d['y']},{d['w']},{d['h']})")
    else:
        print("No tents detected.")

    output_path = image_path.rsplit(".", 1)[0] + "_detected.jpg"
    cv2.imwrite(output_path, processed_frame)
    print(f"Saved annotated image → {output_path}")

    cv2.imshow("Tent Detector", processed_frame)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()