import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def detect_and_straighten(image_path, solution_path, output_size=(500, 500)):
    """
    Detect red and blue corners, visualize detection, and straighten the image.
    Compare result to solution. Detected image is always rotated so that the blue
    corner is at the bottom-left.
    :param image_path: Path to the input image.
    :param solution_path: Path to the solution CSV file.
    :param output_size: Desired size (width, height) for the straightened output.
    """
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert image to HSV for color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for red and blue
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    # Create masks for red and blue
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Morphological operations to remove small detections
    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    
    # Combine masks
    mask_combined = cv2.bitwise_or(mask_red, mask_blue)

    # Find contours for combined mask
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract corner positions using moments
    corners = []
    blue_corner = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Filter out small detections
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:  # Confirm it's within contour
                    corners.append((cx, cy))
                if mask_blue[cy, cx] > 0:
                    blue_corner = (cx, cy)

    # Sort corners: top-left, top-right, bottom-left, bottom-right
    if len(corners) >= 4 and blue_corner:
        corners = sorted(corners, key=lambda x: (x[1], x[0]))
        top_corners = sorted(corners[:2], key=lambda x: x[0])
        bottom_corners = sorted(corners[2:], key=lambda x: x[0])
        sorted_corners = [top_corners[0], top_corners[1], bottom_corners[0], bottom_corners[1]]
        
        # Rearrange corners to ensure blue corner is bottom-left
        if blue_corner != sorted_corners[2]:
            while sorted_corners[2] != blue_corner:
                sorted_corners = sorted_corners[-1:] + sorted_corners[:-1]  # Rotate corners
    else:
        print("Error: Not all corners detected or blue corner not found.")
        return

    # Step-by-Step Visualization
    output_image = image_rgb.copy()
    cv2.drawContours(output_image, contours, -1, (255, 0, 0), 2)  # Draw all contours
    for point in sorted_corners:
        cv2.circle(output_image, point, 10, (0, 255, 0), -1)  # Mark detected corners

    # Perspective Transformation
    width, height = output_size
    destination_corners = np.array([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1]
    ], dtype='float32')

    matrix = cv2.getPerspectiveTransform(np.array(sorted_corners, dtype='float32'), destination_corners)
    straightened_image = cv2.warpPerspective(image_rgb, matrix, (width, height))

    # Generate 5x5 grid of 1s and 0s
    grid = np.zeros((5, 5), dtype=int)
    step_x = width // 5
    step_y = height // 5

    for i in range(5):
        for j in range(5):
            x = int((j + 0.5) * step_x)
            y = int((i + 0.5) * step_y)
            pixel = straightened_image[y, x]
            intensity = np.mean(pixel)
            grid[i, j] = 1 if intensity < 128 else 0

    print("5x5 Grid:")
    print(grid)

    # Load solution and convert '#' cells to 1, others to 0
    solution_raw = pd.read_csv(solution_path, header=None).values
    solution = np.zeros((5, 5), dtype=int)
    cell_numbers = np.zeros((5, 5), dtype=int)

    # Convert solution to binary and store cell numbers
    for i in range(5):
        for j in range(5):
            cell_numbers[i, j] = int(solution_raw[i, j].strip().replace('#', '').strip())  # Extract cell number
            solution[i, j] = 1 if '#' in solution_raw[i, j] else 0  # Mark 1 for '#' and 0 otherwise

    # Compare solution and generated grid
    mismatches = []
    for i in range(5):
        for j in range(5):
            if grid[i, j] != solution[i, j]:
                cell_number = cell_numbers[i, j]  # Calculate the cell number (1-25)
                mismatches.append((cell_number, i + 1, j + 1))  # Cell number, row, column

    if mismatches:
        for cell_number, row, col in mismatches:
            print(f"Mismatch for game {cell_number} (Row {row}, Column {col})")
    else:
        print("Grid matches the solution!")


    # Display results
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)

    plt.subplot(2, 2, 2)
    plt.title("Detected Corners")
    plt.imshow(output_image)

    plt.subplot(2, 2, 3)
    plt.title("Mask: Red and Blue")
    plt.imshow(mask_combined, cmap='gray')

    plt.subplot(2, 2, 4)
    plt.title("Straightened Image")
    plt.imshow(straightened_image)

    plt.tight_layout()
    plt.show()

# Run the function with your image path and solution path
detect_and_straighten('images/image.png', 'solution.csv')