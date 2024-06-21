import cv2
import time
import numpy as np

from bb_utils.screen_capture_mss import ScreenCapture

def main():
    cap = ScreenCapture(debug=True)
    while True:
        frame = cap.read()
        if frame is None:
            print("No frame")
            continue
        
        # Display the frame
        cv2.imshow("Computer Vision", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()