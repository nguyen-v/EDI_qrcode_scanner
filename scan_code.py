import logging
import os
import sys
import time
import psutil  # For memory usage logging
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import math
import pandas as pd
from gpiozero import OutputDevice, Button

# ---------------------------
# Configuration and Constants
# ---------------------------
MATCH_DIST_THRESHOLD = 20      # Maximum distance (in pixels) for matching detections across frames
STABLE_COUNT_THRESHOLD = 2     # Minimum consecutive frames required for a blob to be considered stable
LOST_FRAME_THRESHOLD = 3       # Maximum frames a candidate can be missing before removal
STABLE_DURATION = 1.5          # Duration (in seconds) that the detection must be stable before processing

RELAY_PIN_1 = 20
RELAY_PIN_2 = 21
BUTTON_PIN = 2

relay1 = OutputDevice(RELAY_PIN_1, active_high=True, initial_value=False)
relay2 = OutputDevice(RELAY_PIN_2, active_high=True, initial_value=False)
button = Button(BUTTON_PIN)

# ---------------------------
# Setup Logging
# ---------------------------
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------
# Helper: Crop an image to a central square
# ---------------------------
def crop_center_square(cv_img):
    h, w = cv_img.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    return cv_img[start_y:start_y+min_dim, start_x:start_x+min_dim]

# ---------------------------
# Custom RoundedLabel Class
# ---------------------------
class RoundedLabel(QtWidgets.QLabel):
    def __init__(self, radius=15, parent=None):
        super().__init__(parent)
        self.radius = radius
        self.setStyleSheet("background-color: transparent;")
        self.setScaledContents(False)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        path = QtGui.QPainterPath()
        rectF = QtCore.QRectF(self.rect())
        path.addRoundedRect(rectF, self.radius, self.radius)
        painter.setClipPath(path)
        if self.pixmap():
            pixmap = self.pixmap()
            scaled_pix = pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            painter.drawPixmap(self.rect(), scaled_pix)
        else:
            super().paintEvent(event)
        painter.end()

# ---------------------------
# Helper Functions for Detection
# ---------------------------
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def update_candidates(candidates, detections):
    for cand in candidates:
        cand['matched'] = False
    for (cx, cy) in detections:
        matched = False
        for cand in candidates:
            dist = math.hypot(cand['x'] - cx, cand['y'] - cy)
            if dist < MATCH_DIST_THRESHOLD:
                alpha = 0.7
                cand['x'] = alpha * cx + (1 - alpha) * cand['x']
                cand['y'] = alpha * cy + (1 - alpha) * cand['y']
                cand['count'] += 1
                cand['lost'] = 0
                cand['matched'] = True
                matched = True
                break
        if not matched:
            candidates.append({'x': cx, 'y': cy, 'count': 1, 'lost': 0, 'matched': True})
    for cand in candidates:
        if not cand['matched']:
            cand['lost'] += 1
    candidates[:] = [cand for cand in candidates if cand['lost'] <= LOST_FRAME_THRESHOLD]

# ---------------------------
# Worker Thread for Detection
# ---------------------------
class DetectionWorker(QtCore.QThread):
    update_main_image = QtCore.pyqtSignal(QtGui.QImage)
    update_right_images = QtCore.pyqtSignal(dict)  # Keys: "Original", "Rectified", "Grayscale", "Threshold"
    update_progress = QtCore.pyqtSignal(int)
    show_message = QtCore.pyqtSignal(str)
    detection_complete = QtCore.pyqtSignal(bool, str, QtGui.QImage)  # (success, message, final image)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self.waiting_for_reset = False
        self.reset_mode = "manual"  # "manual" for success, "auto" for mismatch

class DetectionWorker(QtCore.QThread):
    update_main_image = QtCore.pyqtSignal(QtGui.QImage)
    update_right_images = QtCore.pyqtSignal(dict)
    update_progress = QtCore.pyqtSignal(int)
    show_message = QtCore.pyqtSignal(str)
    detection_complete = QtCore.pyqtSignal(bool, str, QtGui.QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self.waiting_for_reset = False
        self.reset_mode = "manual"

    def run(self):
        logger.debug("Worker thread started")
        time.sleep(5)  # Keep the 5-second delay
        
        try:
            logger.debug("Attempting to open camera")
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not cap.isOpened():
                logger.error("Could not open video capture")
                self.show_message.emit("Error: Could not open video capture.")
                return

            red_candidates = []
            blue_candidates = []
            condition_start_time = None
            frame_count = 0

            while self._running:
                logger.debug("Reading frame")
                ret, frame = cap.read()
                if not ret:
                    logger.error("Could not read frame")
                    self.show_message.emit("Error: Could not read frame.")
                    break

                frame_count += 1
                if frame_count % 10 == 0:
                    process = psutil.Process(os.getpid())
                    mem_usage = process.memory_info().rss / (1024 * 1024)  # in MB
                    logger.debug(f"Memory usage: {mem_usage:.2f} MB")

                logger.debug("Checking reset condition")
                if self.waiting_for_reset and self.reset_mode == "manual":
                    logger.debug("Emitting main image for reset")
                    qimg = self.convert_cv_qt(frame)
                    if not qimg.isNull():
                        self.update_main_image.emit(qimg)
                    if button.is_pressed:
                        logger.debug("Button pressed, resetting")
                        relay1.off()
                        relay2.off()
                        self.waiting_for_reset = False
                        condition_start_time = None
                        red_candidates.clear()
                        blue_candidates.clear()
                        self.show_message.emit("")
                        self.update_right_images.emit({})
                        time.sleep(0.5)
                    self.msleep(30)
                    continue

                logger.debug("Converting to HSV")
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                logger.debug("Creating red masks")
                lower_red1 = np.array([0, 140, 30])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 140, 30])
                upper_red2 = np.array([180, 255, 255])
                red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                red_mask = cv2.bitwise_or(red_mask1, red_mask2)

                logger.debug("Creating blue mask")
                lower_blue = np.array([100, 100, 50])
                upper_blue = np.array([140, 255, 255])
                blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

                logger.debug("Finding red contours")
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

                logger.debug("Finding blue contours")
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

                logger.debug("Updating candidates")
                update_candidates(red_candidates, red_detections)
                update_candidates(blue_candidates, blue_detections)

                logger.debug("Drawing markers on frame")
                for cand in red_candidates:
                    if cand['count'] >= STABLE_COUNT_THRESHOLD:
                        cv2.drawMarker(frame, (int(cand['x']), int(cand['y'])), (0, 0, 255),
                                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                for cand in blue_candidates:
                    if cand['count'] >= STABLE_COUNT_THRESHOLD:
                        cv2.drawMarker(frame, (int(cand['x']), int(cand['y'])), (255, 0, 0),
                                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

                logger.debug("Checking stable detections")
                stable_red = [cand for cand in red_candidates if cand['count'] >= STABLE_COUNT_THRESHOLD]
                stable_blue = [cand for cand in blue_candidates if cand['count'] >= STABLE_COUNT_THRESHOLD]
                if len(stable_red) == 3 and len(stable_blue) == 1:
                    if condition_start_time is None:
                        condition_start_time = time.time()
                    elapsed = time.time() - condition_start_time
                    progress = int((elapsed / STABLE_DURATION) * 100)
                    logger.debug(f"Emitting progress: {progress}")
                    self.update_progress.emit(progress)
                    if elapsed >= STABLE_DURATION:
                        logger.debug("Processing stable detection")
                        original = frame.copy()
                        all_candidates = stable_red + stable_blue
                        pts = np.array([(int(cand['x']), int(cand['y'])) for cand in all_candidates], dtype="float32")
                        ordered_pts = order_points(pts)
                        dst_pts = np.array([[0, 0], [499, 0], [499, 499], [0, 499]], dtype="float32")
                        M_persp = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
                        warped = cv2.warpPerspective(frame, M_persp, (500, 500))
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
                        
                        logger.debug("Cropping and matching")
                        original_cropped = crop_center_square(original)
                        success, message, inter_images = self.process_and_match(rotated, original_cropped)
                        logger.debug("Emitting right images")
                        self.update_right_images.emit(inter_images)
                        if success:
                            logger.debug("Success: Activating relays")
                            relay1.on()
                            relay2.on()
                            self.detection_complete.emit(True, message, self.convert_cv_qt(rotated))
                            self.waiting_for_reset = True
                            self.reset_mode = "manual"
                        else:
                            logger.debug("Failure: Emitting detection complete")
                            self.detection_complete.emit(False, message, self.convert_cv_qt(rotated))
                            self.msleep(10000)
                            self.show_message.emit("")
                            self.update_right_images.emit({})
                            condition_start_time = None
                            red_candidates.clear()
                            blue_candidates.clear()
                else:
                    condition_start_time = None
                    logger.debug("Resetting progress")
                    self.update_progress.emit(0)

                logger.debug("Emitting main image")
                qimg = self.convert_cv_qt(frame)
                if not qimg.isNull():
                    self.update_main_image.emit(qimg)
                    logger.debug("Main image emitted successfully")
                else:
                    logger.error("Skipping emission due to null QImage")
                self.msleep(30)

            logger.debug("Releasing camera")
            cap.release()
            logger.debug("Camera released successfully")
        except Exception as e:
            logger.error(f"Exception in worker thread: {str(e)}", exc_info=True)
            self.show_message.emit(f"Worker error: {str(e)}")
        finally:
            logger.debug("Worker thread stopped")

    def convert_cv_qt(self, cv_img, isGray=False, target_size=None):
        try:
            logger.debug("Starting convert_cv_qt")
            if cv_img is None or cv_img.size == 0:
                logger.error("Input cv_img is None or empty")
                return QtGui.QImage()
            
            # Copy the frame to ensure data stability
            cv_img_copy = cv_img.copy()
            
            if isGray:
                qformat = QtGui.QImage.Format_Grayscale8
            else:
                logger.debug("Converting BGR to RGB")
                cv_img_copy = cv2.cvtColor(cv_img_copy, cv2.COLOR_BGR2RGB)
                qformat = QtGui.QImage.Format_RGB888
            
            h, w = cv_img_copy.shape[:2]
            bytes_per_line = w if isGray else 3 * w
            logger.debug(f"Creating QImage: {w}x{h}, bytes_per_line={bytes_per_line}")
            qt_img = QtGui.QImage(cv_img_copy.data, w, h, bytes_per_line, qformat).copy()  # Deep copy
            
            if qt_img.isNull():
                logger.error("Created QImage is null")
                return QtGui.QImage()
            
            if target_size:
                logger.debug(f"Scaling to {target_size}")
                qt_img = qt_img.scaled(target_size[0], target_size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
                if qt_img.isNull():
                    logger.error("Scaled QImage is null")
                    return QtGui.QImage()
            
            logger.debug("convert_cv_qt completed successfully")
            return qt_img
        except Exception as e:
            logger.error(f"Exception in convert_cv_qt: {str(e)}", exc_info=True)
            return QtGui.QImage()

    def process_and_match(self, rotated, original):
        try:
            # Build the 5×5 binary grid from the rectified image
            rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            height, width = rotated.shape[:2]
            grid = np.zeros((5, 5), dtype=int)
            step_x, step_y = width // 5, height // 5

            for i in range(5):
                for j in range(5):
                    x0, y0 = j * step_x, i * step_y
                    cell = rotated_gray[y0:y0+step_y, x0:x0+step_x]
                    _, thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    cy, cx = step_y // 2, step_x // 2
                    grid[i, j] = 1 if thresh[cy, cx] < 128 else 0

            # Determine the absolute path to solution.csv
            script_dir = os.path.dirname(os.path.abspath(__file__))
            solution_path = os.path.join(script_dir, 'solution.csv')
            logger.debug(f"Loading solution.csv from: {solution_path}")

            # Load solution.csv as strings
            solution_df = pd.read_csv(solution_path, header=None, dtype=str)
            solution_bool = np.zeros((5, 5), dtype=int)
            solution_label = np.empty((5, 5), dtype=object)

            for i in range(5):
                for j in range(5):
                    raw = solution_df.iat[i, j].strip()
                    if raw.startswith("#"):
                        solution_bool[i, j] = 1
                        solution_label[i, j] = raw.lstrip("#").strip()
                    else:
                        solution_bool[i, j] = 0
                        solution_label[i, j] = raw

            # Compare and collect mismatches by name
            mismatch_games = { solution_label[i, j]
                            for i in range(5) for j in range(5)
                            if grid[i, j] != solution_bool[i, j] }

            if mismatch_games:
                message = "Erreurs: " + ", ".join(sorted(mismatch_games))
                success = False
            else:
                message = "Le code est correct!"
                success = True

            # Build intermediate QImages
            orig_qimg = self.convert_cv_qt(original, target_size=(350, 350))
            rect_qimg = self.convert_cv_qt(rotated, target_size=(350, 350))
            gray_qimg = self.convert_cv_qt(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY),
                                        isGray=True, target_size=(350, 350))
            _, thresh_img = cv2.threshold(rotated_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh_qimg = self.convert_cv_qt(thresh_img, isGray=True, target_size=(350, 350))

            inter_images = {
                "Original": orig_qimg,
                "Rectified": rect_qimg,
                "Grayscale": gray_qimg,
                "Threshold": thresh_qimg
            }

            return success, message, inter_images
        except Exception as e:
            logger.error(f"Exception in process_and_match: {str(e)}", exc_info=True)
            return False, f"Processing error: {str(e)}", {}

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
        self.resize(1200, 700)
        self.initUI()
        self.setCursor(QtGui.QCursor(QtCore.Qt.BlankCursor))

    def showEvent(self, event):
        super().showEvent(event)
        if not hasattr(self, "_fullscreened"):
            self.setWindowState(self.windowState() | Qt.WindowFullScreen)  # Still disabled for debugging
            self._fullscreened = True
            QtCore.QTimer.singleShot(1000, self.start_worker)  # Delay by 1 second

    def start_worker(self):
        self.worker = DetectionWorker()
        self.worker.update_main_image.connect(self.setMainImage)
        self.worker.update_right_images.connect(self.updateRightPanel)
        self.worker.update_progress.connect(self.setProgress)
        self.worker.show_message.connect(self.showMessage)
        self.worker.detection_complete.connect(self.handleDetectionComplete)
        self.worker.start()

    def initUI(self):
        dark_stylesheet = """
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
            font-family: 'Unytour';  /* Set global font to Unytour */
            font-size: 14px;
        }
        QFrame#rightFrame {
            border: none;
            background-color: #3a3a3a;
        }
        QLabel#rightMessage {
            font-family: 'Unytour';  /* Explicitly set for rightMessage */
            font-size: 20px;
            font-weight: bold;
        }
        QProgressBar {
            border: 2px solid #555;
            border-radius: 5px;
            text-align: center;
            font-family: 'Unytour';  /* Set for progress bar text */
        }
        QProgressBar::chunk {
            background-color: #3d8ec9;
        }
        QLabel {
            font-family: 'Unytour';  /* Ensure all labels use Unytour */
        }
        """
        self.setStyleSheet(dark_stylesheet)
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left panel: camera feed with progress bar below
        left_frame = QtWidgets.QFrame()
        left_layout = QtWidgets.QVBoxLayout(left_frame)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        self.main_image_label = RoundedLabel(radius=15)
        self.main_image_label.setMinimumSize(600, 400)
        left_layout.addWidget(self.main_image_label)
        self.left_progress_bar = QtWidgets.QProgressBar()
        self.left_progress_bar.setValue(0)
        left_layout.addWidget(self.left_progress_bar)
        main_layout.addWidget(left_frame, 3)

        # Right panel: intermediate images with titles
        right_frame = QtWidgets.QFrame()
        right_frame.setObjectName("rightFrame")
        right_frame.setMaximumWidth(750)
        right_layout = QtWidgets.QVBoxLayout(right_frame)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)
        grid_widget = QtWidgets.QWidget()
        grid_layout = QtWidgets.QGridLayout(grid_widget)
        grid_layout.setContentsMargins(5, 5, 5, 5)
        grid_layout.setSpacing(10)
        # Container for Original image
        self.containerOriginal = QtWidgets.QWidget()
        layoutOrig = QtWidgets.QVBoxLayout(self.containerOriginal)
        layoutOrig.setContentsMargins(0, 0, 0, 0)
        layoutOrig.setSpacing(5)
        self.original_title = QtWidgets.QLabel("Original")
        self.original_title.setAlignment(Qt.AlignCenter)
        self.original_title.setStyleSheet("font-size: 32px; font-weight: bold;")
        self.original_image = RoundedLabel(radius=15)
        self.original_image.setFixedSize(350, 350)
        layoutOrig.addWidget(self.original_title)
        layoutOrig.addWidget(self.original_image)
        # Container for Rectified image
        self.containerRectified = QtWidgets.QWidget()
        layoutRect = QtWidgets.QVBoxLayout(self.containerRectified)
        layoutRect.setContentsMargins(0, 0, 0, 0)
        layoutRect.setSpacing(5)
        self.rectified_title = QtWidgets.QLabel("Rectifié")
        self.rectified_title.setAlignment(Qt.AlignCenter)
        self.rectified_title.setStyleSheet("font-size: 32px; font-weight: bold;")
        self.rectified_image = RoundedLabel(radius=15)
        self.rectified_image.setFixedSize(350, 350)
        layoutRect.addWidget(self.rectified_title)
        layoutRect.addWidget(self.rectified_image)
        # Container for Gray image
        self.containerGray = QtWidgets.QWidget()
        layoutGray = QtWidgets.QVBoxLayout(self.containerGray)
        layoutGray.setContentsMargins(0, 0, 0, 0)
        layoutGray.setSpacing(5)
        self.gray_title = QtWidgets.QLabel("Noir/Blanc")
        self.gray_title.setAlignment(Qt.AlignCenter)
        self.gray_title.setStyleSheet("font-size: 32px; font-weight: bold;")
        self.gray_image = RoundedLabel(radius=15)
        self.gray_image.setFixedSize(350, 350)
        layoutGray.addWidget(self.gray_title)
        layoutGray.addWidget(self.gray_image)
        # Container for Thresholded image
        self.containerThresholded = QtWidgets.QWidget()
        layoutThresh = QtWidgets.QVBoxLayout(self.containerThresholded)
        layoutThresh.setContentsMargins(0, 0, 0, 0)
        layoutThresh.setSpacing(5)
        self.threshold_title = QtWidgets.QLabel("Seuillage")
        self.threshold_title.setAlignment(Qt.AlignCenter)
        self.threshold_title.setStyleSheet("font-size: 32px; font-weight: bold;")
        self.threshold_image = RoundedLabel(radius=15)
        self.threshold_image.setFixedSize(350, 350)
        layoutThresh.addWidget(self.threshold_title)
        layoutThresh.addWidget(self.threshold_image)

        self.original_title.hide()
        self.rectified_title.hide()
        self.gray_title.hide()
        self.threshold_title.hide()

        # Add containers to the grid
        grid_layout.addWidget(self.containerOriginal, 0, 0)
        grid_layout.addWidget(self.containerRectified, 0, 1)
        grid_layout.addWidget(self.containerGray, 1, 0)
        grid_layout.addWidget(self.containerThresholded, 1, 1)
        right_layout.addWidget(grid_widget)
        self.right_message_label = QtWidgets.QLabel("")
        self.right_message_label.setObjectName("rightMessage")
        self.right_message_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.right_message_label)
        main_layout.addWidget(right_frame, 1)

    def setMainImage(self, qimg):
        try:
            pixmap = QtGui.QPixmap.fromImage(qimg)
            scaled = pixmap.scaled(self.main_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.main_image_label.setPixmap(scaled)
        except Exception as e:
            logger.error(f"Exception in setMainImage: {str(e)}", exc_info=True)

    def updateRightPanel(self, images):
        try:
            if not images:
                self.original_image.clear()
                self.rectified_image.clear()
                self.gray_image.clear()
                self.threshold_image.clear()
                self.original_title.hide()
                self.rectified_title.hide()
                self.gray_title.hide()
                self.threshold_title.hide()
                return
            self.original_title.show()
            self.rectified_title.show()
            self.gray_title.show()
            self.threshold_title.show()
            if "Original" in images:
                self.original_image.setPixmap(QtGui.QPixmap.fromImage(images["Original"]))
            if "Rectified" in images:
                self.rectified_image.setPixmap(QtGui.QPixmap.fromImage(images["Rectified"]))
            if "Grayscale" in images:
                self.gray_image.setPixmap(QtGui.QPixmap.fromImage(images["Grayscale"]))
            if "Threshold" in images:
                self.threshold_image.setPixmap(QtGui.QPixmap.fromImage(images["Threshold"]))
        except Exception as e:
            logger.error(f"Exception in updateRightPanel: {str(e)}", exc_info=True)

    def setProgress(self, value):
        try:
            self.left_progress_bar.setValue(value)
            # Show "Analyse en cours..." when progress is between 1 and 99
            if 0 < value < 100:
                self.right_message_label.setText("Analyse en cours...")
                self.right_message_label.setStyleSheet("color: #ffffff; font-size: 20px; font-weight: bold;")
            elif value == 0:
                self.right_message_label.setText("")  # Clear the message when not analyzing
                self.right_message_label.setStyleSheet("color: #ffffff; font-size: 20px; font-weight: bold;")
        except Exception as e:
            logger.error(f"Exception in setProgress: {str(e)}", exc_info=True)

    def showMessage(self, message):
        try:
            self.right_message_label.setText(message)
        except Exception as e:
            logger.error(f"Exception in showMessage: {str(e)}", exc_info=True)

    def handleDetectionComplete(self, success, message, final_qimg):
        try:
            self.setMainImage(final_qimg)
            if success:
                display_text = "Code correct détecté. Ouverture du coffre..."
                color = "#90EE90"
            else:
                display_text = message
                color = "red"
            self.right_message_label.setText(display_text)
            self.right_message_label.setStyleSheet(f"color: {color};")
            self.left_progress_bar.setValue(0)
        except Exception as e:
            logger.error(f"Exception in handleDetectionComplete: {str(e)}", exc_info=True)

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

# ---------------------------
# Application Entry Point
# ---------------------------
if __name__ == "__main__":
    from PyQt5.QtCore import QCoreApplication, Qt
    QCoreApplication.setAttribute(Qt.AA_X11InitThreads)
    # QCoreApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)  # Disabled for debugging

    import os
    os.environ["QT_QPA_PLATFORM"] = "xcb"

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())