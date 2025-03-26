import cv2
import numpy as np
import math
import time
import pandas as pd
from gpiozero import OutputDevice, Button

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

# ---------------------------
# Configuration and Constants
# ---------------------------
MATCH_DIST_THRESHOLD = 20      # Maximum distance (in pixels) for matching detections across frames
STABLE_COUNT_THRESHOLD = 2     # Minimum consecutive frames required for a blob to be considered stable
LOST_FRAME_THRESHOLD = 3       # Maximum frames a candidate can be missing before removal
STABLE_DURATION = 3.0          # Duration (in seconds) that the detection must be stable before processing

RELAY_PIN_1 = 20
RELAY_PIN_2 = 21
BUTTON_PIN = 2

relay1 = OutputDevice(RELAY_PIN_1, active_high=True, initial_value=False)
relay2 = OutputDevice(RELAY_PIN_2, active_high=True, initial_value=False)
button = Button(BUTTON_PIN)

# ---------------------------
# Helper Functions
# ---------------------------
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
                alpha = 0.7  # responsiveness factor
                cand['x'] = alpha * cx + (1 - alpha) * cand['x']
                cand['y'] = alpha * cy + (1 - alpha) * cand['y']
                cand['count'] += 1
                cand['lost'] = 0
                cand['matched'] = True
                matched = True
                break
        if not matched:
            candidates.append({'x': cx, 'y': cy, 'count': 1, 'lost': 0, 'matched': True})
    
    # Increase lost counter for unmatched candidates and remove stale ones
    for cand in candidates:
        if not cand['matched']:
            cand['lost'] += 1
    candidates[:] = [cand for cand in candidates if cand['lost'] <= LOST_FRAME_THRESHOLD]

# ---------------------------
# Worker Thread for Detection
# ---------------------------
class DetectionWorker(QtCore.QThread):
    # Signals to update the UI
    update_main_image = QtCore.pyqtSignal(QtGui.QImage)
    update_right_images = QtCore.pyqtSignal(dict)  # Keys: "Original", "Grayscale", "Threshold", "Rectified"
    update_progress = QtCore.pyqtSignal(int)
    show_message = QtCore.pyqtSignal(str)
    detection_complete = QtCore.pyqtSignal(bool, str, QtGui.QImage)  # (success, message, final image)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        # Flag to indicate we are waiting for a manual reset (only used after a successful detection)
        self.waiting_for_reset = False
        self.reset_mode = "manual"  # "manual" for success, "auto" for mismatch

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.show_message.emit("Error: Could not open video capture.")
            return

        red_candidates = []
        blue_candidates = []
        condition_start_time = None

        while self._running:
            ret, frame = cap.read()
            if not ret:
                self.show_message.emit("Error: Could not read frame.")
                break

            # If waiting for manual reset after a successful detection, check button press.
            if self.waiting_for_reset and self.reset_mode == "manual":
                # Do not overlay any text on the camera feed.
                self.update_main_image.emit(self.convert_cv_qt(frame))
                if button.is_pressed:
                    relay1.off()
                    relay2.off()
                    self.waiting_for_reset = False
                    self.reset_mode = "manual"
                    condition_start_time = None
                    red_candidates.clear()
                    blue_candidates.clear()
                    # Clear message and right panel images
                    self.show_message.emit("")
                    self.update_right_images.emit({})
                    time.sleep(0.5)  # debounce delay
                self.msleep(30)
                continue

            # --- Process frame for color detection ---
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Red detection (two ranges)
            lower_red1 = np.array([0, 140, 30])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 140, 30])
            upper_red2 = np.array([180, 255, 255])
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            # Blue detection
            lower_blue = np.array([100, 100, 50])
            upper_blue = np.array([140, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # --- Find contours and centroids for red blobs ---
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
            # --- Find blue blobs ---
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

            # --- Update candidate lists ---
            update_candidates(red_candidates, red_detections)
            update_candidates(blue_candidates, blue_detections)

            # --- Draw markers for stable candidates ---
            for cand in red_candidates:
                if cand['count'] >= STABLE_COUNT_THRESHOLD:
                    cv2.drawMarker(frame, (int(cand['x']), int(cand['y'])), (0, 0, 255),
                                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            for cand in blue_candidates:
                if cand['count'] >= STABLE_COUNT_THRESHOLD:
                    cv2.drawMarker(frame, (int(cand['x']), int(cand['y'])), (255, 0, 0),
                                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            # --- Check for exactly 3 stable red and 1 stable blue blobs ---
            stable_red = [cand for cand in red_candidates if cand['count'] >= STABLE_COUNT_THRESHOLD]
            stable_blue = [cand for cand in blue_candidates if cand['count'] >= STABLE_COUNT_THRESHOLD]
            if len(stable_red) == 3 and len(stable_blue) == 1:
                if condition_start_time is None:
                    condition_start_time = time.time()
                elapsed = time.time() - condition_start_time
                progress = int((elapsed / STABLE_DURATION) * 100)
                self.update_progress.emit(progress)
                if elapsed >= STABLE_DURATION:
                    # Capture the original frame (before rectification)
                    original = frame.copy()
                    # Perspective transform
                    all_candidates = stable_red + stable_blue
                    pts = np.array([(int(cand['x']), int(cand['y'])) for cand in all_candidates], dtype="float32")
                    ordered_pts = order_points(pts)
                    dst_pts = np.array([[0, 0], [499, 0], [499, 499], [0, 499]], dtype="float32")
                    M_persp = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
                    warped = cv2.warpPerspective(frame, M_persp, (500, 500))
                    # Rotate so that the blue candidate is at bottom-left
                    blue_pt = np.array([int(stable_blue[0]['x']), int(stable_blue[0]['y'])], dtype="float32")
                    distances = [np.linalg.norm(blue_pt - pt) for pt in ordered_pts]
                    blue_index = int(np.argmin(distances))
                    rotations_needed = (3 - blue_index) % 4
                    rotated = warped.copy()
                    if rotations_needed == 1:
                        rotated = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                    elif rotations_needed == 2:
                        rotated = cv2.rotate(warped, cv2.ROTATE_180)
                    elif rotations_needed == 3:
                        rotated = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                    # Process and match the rotated image (using the original as well)
                    success, message, inter_images = self.process_and_match(rotated, original)
                    self.update_right_images.emit(inter_images)
                    if success:
                        relay1.on()
                        relay2.on()
                        self.detection_complete.emit(True, message, self.convert_cv_qt(rotated))
                        # Switch to manual reset mode (wait for physical button press)
                        self.waiting_for_reset = True
                        self.reset_mode = "manual"
                    else:
                        self.detection_complete.emit(False, message, self.convert_cv_qt(rotated))
                        # Show mismatch for 10 seconds then auto-reset
                        time.sleep(10)
                        self.show_message.emit("")
                        self.update_right_images.emit({})
                        condition_start_time = None
                        red_candidates.clear()
                        blue_candidates.clear()
            else:
                condition_start_time = None
                self.update_progress.emit(0)

            # Update the main (left) panel with the (possibly annotated) camera feed.
            self.update_main_image.emit(self.convert_cv_qt(frame))
            self.msleep(30)
        cap.release()

    def process_and_match(self, rotated, original):
        # Compute grid from the rectified (rotated) image.
        rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        height, width = rotated.shape[:2]
        grid = np.zeros((5, 5), dtype=int)
        step_x = width // 5
        step_y = height // 5

        for i in range(5):
            for j in range(5):
                x_start = j * step_x
                y_start = i * step_y
                x_end = x_start + step_x
                y_end = y_start + step_y
                cell_region = rotated_gray[y_start:y_end, x_start:x_end]
                _, cell_thresh = cv2.threshold(cell_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                center_y = (y_end - y_start) // 2
                center_x = (x_end - x_start) // 2
                pixel = cell_thresh[center_y, center_x]
                grid[i, j] = 1 if pixel < 128 else 0

        # Load solution CSV and create solution grid.
        solution_path = 'solution.csv'
        solution_raw = pd.read_csv(solution_path, header=None).values
        solution = np.zeros((5, 5), dtype=int)
        cell_numbers = np.zeros((5, 5), dtype=int)
        for i in range(5):
            for j in range(5):
                cell_str = str(solution_raw[i, j]).strip()
                cell_numbers[i, j] = int(cell_str.replace('#', '').strip())
                solution[i, j] = 1 if '#' in cell_str else 0

        mismatches = []
        for i in range(5):
            for j in range(5):
                if grid[i, j] != solution[i, j]:
                    cell_number = cell_numbers[i, j]
                    mismatches.append((cell_number, i + 1, j + 1))
        if mismatches:
            message = ""
            for cell_number, row, col in mismatches:
                message += f"Mismatch for game {cell_number} (Row {row}, Col {col})\n"
            success = False
        else:
            message = "Grid matches solution!"
            success = True

        # Prepare intermediate images (scaled to 350x350 for a bigger display).
        orig_qimg = self.convert_cv_qt(original, target_size=(350, 350))
        rectified_qimg = self.convert_cv_qt(rotated, target_size=(350, 350))
        gray_qimg = self.convert_cv_qt(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY), isGray=True, target_size=(350, 350))
        _, thresh_img = cv2.threshold(rotated_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_qimg = self.convert_cv_qt(thresh_img, isGray=True, target_size=(350, 350))
        inter_images = {
            "Original": orig_qimg,
            "Grayscale": gray_qimg,
            "Threshold": thresh_qimg,
            "Rectified": rectified_qimg
        }
        return success, message, inter_images

    def convert_cv_qt(self, cv_img, isGray=False, target_size=None):
        """Convert from an OpenCV image to QImage and scale if target_size is provided."""
        if isGray:
            qformat = QtGui.QImage.Format_Grayscale8
        else:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            qformat = QtGui.QImage.Format_RGB888
        h, w = cv_img.shape[:2]
        bytes_per_line = w if isGray else 3 * w
        qt_img = QtGui.QImage(cv_img.data, w, h, bytes_per_line, qformat)
        if target_size:
            qt_img = qt_img.scaled(target_size[0], target_size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return qt_img

    def stop(self):
        self._running = False
        self.wait()

# ---------------------------
# Main Window with PyQt5 Interface
# ---------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Detection Interface")
        self.setGeometry(100, 100, 1200, 600)
        self.initUI()
        self.worker = DetectionWorker()
        # Connect worker signals to slots.
        self.worker.update_main_image.connect(self.setMainImage)
        self.worker.update_right_images.connect(self.updateRightPanel)
        self.worker.update_progress.connect(self.setProgress)
        self.worker.show_message.connect(self.showMessage)
        self.worker.detection_complete.connect(self.handleDetectionComplete)
        self.worker.start()

    def initUI(self):
        # Dark mode style with larger text on the right.
        dark_stylesheet = """
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
            font-size: 14px;
        }
        QLabel#rightMessage {
            font-size: 20px;
            font-weight: bold;
        }
        QProgressBar {
            border: 2px solid #555;
            border-radius: 5px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #3d8ec9;
        }
        """
        self.setStyleSheet(dark_stylesheet)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left panel: camera feed (expanding) with progress bar below.
        left_panel = QtWidgets.QVBoxLayout()
        self.main_image_label = QtWidgets.QLabel()
        self.main_image_label.setAlignment(Qt.AlignCenter)
        self.main_image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        left_panel.addWidget(self.main_image_label)
        self.left_progress_bar = QtWidgets.QProgressBar()
        self.left_progress_bar.setValue(0)
        left_panel.addWidget(self.left_progress_bar)
        main_layout.addLayout(left_panel, 1)

        # Right panel: 2x2 grid for intermediate images and a centered message at the bottom.
        right_panel = QtWidgets.QVBoxLayout()
        grid_widget = QtWidgets.QWidget()
        grid_layout = QtWidgets.QGridLayout()
        grid_widget.setLayout(grid_layout)
        # Create labels for intermediate images.
        self.original_label = QtWidgets.QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.grayscale_label = QtWidgets.QLabel()
        self.grayscale_label.setAlignment(Qt.AlignCenter)
        self.threshold_label = QtWidgets.QLabel()
        self.threshold_label.setAlignment(Qt.AlignCenter)
        self.rectified_label = QtWidgets.QLabel()
        self.rectified_label.setAlignment(Qt.AlignCenter)
        grid_layout.addWidget(self.original_label, 0, 0)
        grid_layout.addWidget(self.grayscale_label, 0, 1)
        grid_layout.addWidget(self.threshold_label, 1, 0)
        grid_layout.addWidget(self.rectified_label, 1, 1)
        right_panel.addWidget(grid_widget)
        # Message label at the bottom of the right panel.
        self.right_message_label = QtWidgets.QLabel("")
        self.right_message_label.setObjectName("rightMessage")
        self.right_message_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.right_message_label)
        main_layout.addLayout(right_panel, 1)

    def setMainImage(self, qimg):
        # Scale the incoming image to the current size of the main_image_label.
        pixmap = QtGui.QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.main_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.main_image_label.setPixmap(scaled)

    def updateRightPanel(self, images):
        # If the dictionary is empty, clear the images.
        if not images:
            self.original_label.clear()
            self.grayscale_label.clear()
            self.threshold_label.clear()
            self.rectified_label.clear()
            return
        if "Original" in images:
            self.original_label.setPixmap(QtGui.QPixmap.fromImage(images["Original"]))
        if "Grayscale" in images:
            self.grayscale_label.setPixmap(QtGui.QPixmap.fromImage(images["Grayscale"]))
        if "Threshold" in images:
            self.threshold_label.setPixmap(QtGui.QPixmap.fromImage(images["Threshold"]))
        if "Rectified" in images:
            self.rectified_label.setPixmap(QtGui.QPixmap.fromImage(images["Rectified"]))

    def setProgress(self, value):
        self.left_progress_bar.setValue(value)

    def showMessage(self, message):
        self.right_message_label.setText(message)

    def handleDetectionComplete(self, success, message, final_qimg):
        self.setMainImage(final_qimg)
        self.right_message_label.setText(message)
        self.left_progress_bar.setValue(0)

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

# ---------------------------
# Application Entry Point
# ---------------------------
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
