from qreader import QReader
import cv2


# Create a QReader instance
qreader = QReader()

# Get the image that contains the QR code (QReader expects an uint8 numpy array)
image = cv2.imread("images/qr3.png")

# Use the detect_and_decode function to get the decoded QR data
decoded_text = qreader.detect_and_decode(image=image)
print(decoded_text)