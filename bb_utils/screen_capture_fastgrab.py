import cv2
import time
from fastgrab import screenshot


t0 = time.time()
n_frames = 1
while True:
    img = screenshot.Screenshot().capture()

    cv2.imshow("Computer Vision", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    elapsed_time = time.time() - t0
    avg_fps = (n_frames / elapsed_time)
    print("Average FPS: " + str(avg_fps))
    n_frames += 1