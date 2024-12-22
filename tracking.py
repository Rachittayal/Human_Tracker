import cv2
import numpy as np

# Initialize the webcam feed
cap = cv2.VideoCapture(0)

# Initialize the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Set up the window to display the feed
cv2.namedWindow("Human Movement Tracker")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction to get the foreground mask
    fgmask = fgbg.apply(gray)

    # Find contours of the moving object
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small movements
            # Get the bounding box for the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Draw the bounding box around the moving human
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the original frame with bounding boxes
    cv2.imshow("Human Movement Tracker", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
