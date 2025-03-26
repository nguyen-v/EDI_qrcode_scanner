import cv2
import numpy as np
import math
import time
import pandas as pd
from gpiozero import OutputDevice, Button

# Parameters for temporal filtering
MATCH_DIST_THRESHOLD = 20      # Maximum distance (in pixels) for matching detections across frames
STABLE_COUNT_THRESHOLD = 2     # Minimum consecutive frames required for a blob to be considered stable
LOST_FRAME_THRESHOLD = 3       # Maximum frames a candidate can be missing before removal

# Duration (in seconds) that the detection must be stable before processing the image
STABLE_DURATION = 3.0

RELAY_PIN_1 = 20
RELAY_PIN_2 = 21
BUTTON_PIN = 2

relay1 = OutputDevice(RELAY_PIN_1, active_high=True, initial_value=False)
relay2 = OutputDevice(RELAY_PIN_2, active_high=True, initial_value=False)
button = Button(BUTTON_PIN)

def order_points(pts):
    # Returns points ordered as [top-left, top-right, bottom-right, bottom-left]
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def update_candidates(candidates, detections):
    # Mark all current candidates as unmatched
    for cand in candidates:
        cand['matched'] = False

    # Try to match each detection with an existing candidate
    for (cx, cy) in detections:
        matched = False
        for cand in candidates:
            dist = math.hypot(cand['x'] - cx, cand['y'] - cy)
            if dist < MATCH_DIST_THRESHOLD:
                alpha = 0.7  # Adjust alpha between 0 (no responsiveness) and 1 (fully responsive)
                cand['x'] = alpha * cx + (1 - alpha) * cand['x']
                cand['y'] = alpha * cy + (1 - alpha) * cand['y']
                cand['count'] += 1
                cand['lost'] = 0
                cand['matched'] = True
                matched = True
                break
        if not matched:
            # Create a new candidate if no match was found
            candidates.append({'x': cx, 'y': cy, 'count': 1, 'lost': 0, 'matched': True})
    
    # For candidates not matched this frame, increase the lost counter
    for cand in candidates:
        if not cand['matched']:
            cand['lost'] += 1

    # Remove candidates that have been missing for too many frames
    candidates[:] = [cand for cand in candidates if cand['lost'] <= LOST_FRAME_THRESHOLD]

def process_and_match(rotated):
    # Convert to grayscale
    rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

    height, width = rotated.shape[:2]
    grid = np.zeros((5, 5), dtype=int)
    step_x = width // 5
    step_y = height // 5

    # --- Local thresholding per grid cell ---
    for i in range(5):
        for j in range(5):
            x_start = j * step_x
            y_start = i * step_y
            x_end = x_start + step_x
            y_end = y_start + step_y

            cell_region = rotated_gray[y_start:y_end, x_start:x_end]
            
            # Use Otsu's thresholding locally on each cell
            _, cell_thresh = cv2.threshold(cell_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Sample the center pixel of the thresholded cell
            center_y = (y_end - y_start) // 2
            center_x = (x_end - x_start) // 2
            pixel = cell_thresh[center_y, center_x]

            grid[i, j] = 1 if pixel < 128 else 0

    print("Detected grid from processed image:")
    print(grid)

    # --- Load the solution CSV and create the solution grid ---
    solution_path = 'solution.csv'
    solution_raw = pd.read_csv(solution_path, header=None).values
    solution = np.zeros((5, 5), dtype=int)
    cell_numbers = np.zeros((5, 5), dtype=int)
    for i in range(5):
        for j in range(5):
            cell_str = str(solution_raw[i, j]).strip()
            cell_numbers[i, j] = int(cell_str.replace('#', '').strip())
            solution[i, j] = 1 if '#' in cell_str else 0

    print("Solution grid from CSV:")
    print(solution)

    # --- Compare the generated grid with the solution ---
    mismatches = []
    for i in range(5):
        for j in range(5):
            if grid[i, j] != solution[i, j]:
                cell_number = cell_numbers[i, j]
                mismatches.append((cell_number, i + 1, j + 1))

    if mismatches:
        for cell_number, row, col in mismatches:
            print(f"Mismatch for game {cell_number} (Row {row}, Column {col})")
    else:
        print("Grid matches the solution!")
        relay1.on()
        relay2.on()

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    red_candidates = []
    blue_candidates = []
    
    # Variable to track when the condition became stable
    condition_start_time = None

    # Flag to block new detection until the button is pressed.
    waiting_for_button = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # If we're waiting for the user to press the button to reset, skip detection.
        if waiting_for_button:
            cv2.putText(frame, "Press button to reset", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Camera Feed", frame)
            if button.is_pressed:
                # Button press resets the system and turns off the relays.
                relay1.off()
                relay2.off()
                waiting_for_button = False
                condition_start_time = None
                red_candidates.clear()
                blue_candidates.clear()
                print("System reset. Ready for new detection.")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Convert frame to HSV and detect colors
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # --- Detecting Red Blobs ---
        lower_red1 = np.array([0, 140, 30])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 140, 30])
        upper_red2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # --- Detecting Blue Blobs ---
        lower_blue = np.array([100, 100, 50])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Find red detections that meet the area threshold
        red_detections = []
        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_red:
            area = cv2.contourArea(cnt)
            if 100 < area < 400:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    red_detections.append((cx, cy))
        
        # Find blue detections that meet the area threshold
        blue_detections = []
        contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_blue:
            area = cv2.contourArea(cnt)
            if 50 < area < 2000:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    blue_detections.append((cx, cy))
        
        # Update candidate lists with current detections
        update_candidates(red_candidates, red_detections)
        update_candidates(blue_candidates, blue_detections)
        
        # Draw stable red candidates
        for cand in red_candidates:
            if cand['count'] >= STABLE_COUNT_THRESHOLD:
                cv2.drawMarker(frame, (int(cand['x']), int(cand['y'])), (0, 0, 255),
                               markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        
        # Draw stable blue candidates
        for cand in blue_candidates:
            if cand['count'] >= STABLE_COUNT_THRESHOLD:
                cv2.drawMarker(frame, (int(cand['x']), int(cand['y'])), (255, 0, 0),
                               markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        
        # Check for exactly 3 stable red blobs and exactly 1 stable blue blob
        stable_red = [cand for cand in red_candidates if cand['count'] >= STABLE_COUNT_THRESHOLD]
        stable_blue = [cand for cand in blue_candidates if cand['count'] >= STABLE_COUNT_THRESHOLD]
        if len(stable_red) == 3 and len(stable_blue) == 1:
            if condition_start_time is None:
                condition_start_time = time.time()
            elapsed = time.time() - condition_start_time
            remaining = max(0, int(STABLE_DURATION - elapsed) + 1)
            cv2.putText(frame, f"Hold for {remaining}s", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Once the condition is stable for the required duration, process the image
            if elapsed >= STABLE_DURATION:
                # Combine candidate positions (3 red + 1 blue = 4 points)
                all_candidates = stable_red + stable_blue
                pts = np.array([(int(cand['x']), int(cand['y'])) for cand in all_candidates], dtype="float32")
                # Order the points for a consistent quadrilateral
                ordered_pts = order_points(pts)
                # Destination points for a 500x500 square
                dst_pts = np.array([[0, 0], [499, 0], [499, 499], [0, 499]], dtype="float32")
                # Compute perspective transform and warp the frame
                M_persp = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
                warped = cv2.warpPerspective(frame, M_persp, (500, 500))
                
                # --- Rotate the warped image so that the blue candidate appears at the bottom left ---
                blue_pt = np.array([int(stable_blue[0]['x']), int(stable_blue[0]['y'])], dtype="float32")
                distances = [np.linalg.norm(blue_pt - pt) for pt in ordered_pts]
                blue_index = int(np.argmin(distances))
                # We want the blue candidate to be at index 3 (bottom-left)
                rotations_needed = (3 - blue_index) % 4
                rotated = warped
                if rotations_needed == 1:
                    rotated = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                elif rotations_needed == 2:
                    rotated = cv2.rotate(warped, cv2.ROTATE_180)
                elif rotations_needed == 3:
                    rotated = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # --- Match the processed image (rotated) against the solution ---
                process_and_match(rotated)
                
                # Set flag to wait for the user to press the button before restarting detection.
                waiting_for_button = True

        else:
            # Reset the timer if the condition is not continuously met.
            condition_start_time = None

        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
