import cv2
import numpy as np
import math

# dict for tent colours in HSV colour space
# Hue range: 0-180, Saturation range: 0-255, Value range: 0-255
# Hue is the colour type, Saturation is the intensity of the colour, and Value is the brightness of the colour
TENT_COLOURS = {
    'Blue': [np.array([100, 150, 50]), np.array([130, 255, 255])],
    'Neon Green': [np.array([35, 100, 100]), np.array([85, 255, 255])],
    'White': [np.array([0, 0, 200]), np.array([180, 30, 255])],
    'Orange': [np.array([5, 150, 150]), np.array([15, 255, 255])]
}

def detect_tent(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, (lower_bound, upper_bound) in TENT_COLOURS.items():
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            print(area)
            if 500 < area < 10000:
                perimeter = cv2.arcLength(contour, True)
                print(perimeter)
                if perimeter == 0:
                    continue
                circularity = (4 * math.pi * area) / (perimeter * perimeter)
                print(circularity)
                if 0.5 < circularity < 1.0:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{color_name} Tent", (x, y-10), cv2.FONT_ITALIC, 0.6, (0,255,0), 2)

    return frame

if __name__ == "__main__":
    image_path = "test_media/test3.jpg" 
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not load image at {image_path}. Check the file path!")
    else:
        height, width = frame.shape[:2]
        if width > 1280:
            scale = 1280 / width
            frame = cv2.resize(frame, (1280, int(height * scale)))
        
        processed_frame = detect_tent(frame)
        
        cv2.imshow("Tent Detector - Single Image", processed_frame)
        print("Success! Press any key on the image window to close it...")
        
        cv2.waitKey(0) 
        cv2.destroyAllWindows()      