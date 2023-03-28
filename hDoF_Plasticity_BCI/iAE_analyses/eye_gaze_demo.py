# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 12:10:11 2023

@author: nikic
"""

import cv2

# Create a VideoCapture object to read from the webcam
cap = cv2.VideoCapture(0)

# Set the resolution of the video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create a CascadeClassifier object to detect eyes
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Loop to read frames from the webcam and process them
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Convert the frame to grayscale for better detection performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect eyes in the grayscale image using the eye cascade classifier
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Loop through the detected eyes and draw rectangles around them
    for (x,y,w,h) in eyes:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
        # Calculate the size of the pupil by measuring the area of the eye region
        eye_region = gray[y:y+h, x:x+w]
        _, thresh = cv2.threshold(eye_region, 30, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            (pupil_x, pupil_y, pupil_w, pupil_h) = cv2.boundingRect(max_contour)
            pupil_area = cv2.contourArea(max_contour)
            cv2.rectangle(frame, (x+pupil_x, y+pupil_y), (x+pupil_x+pupil_w, y+pupil_y+pupil_h), (255, 0, 0), 2)
            
            # Print the size of the pupil to the console
            print(f"Pupil area: {pupil_area}")
    
    # Show the frame with rectangles drawn around the eyes and pupils
    cv2.imshow("Frame", frame)
    
    # Wait for a key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
