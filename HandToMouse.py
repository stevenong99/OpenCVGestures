import cv2
import time
import mediapipe as mp
import HandTrackingModule as htm
import PyMouse as mouse
import math
import numpy as np

if __name__ == "__main__":
    scr_width, scr_height = 1920, 1080
    cam_width, cam_height = 640, 480
    smooth = 2
    ploc_x , ploc_y = 0, 0
    cloc_x , cloc_y = 0, 0
    prev_click = 0
    prev_scroll = 0


    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, cam_width)
    cap.set(4, cam_height)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    start = 0
    current = 0

    hd = htm.Hand_Detector(detection_confidence=0.7, tracking_confidence=0.5)

    while True:
        # Reading frame by frame
        ret, frame = cap.read()

        # Flipping the frame
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # If frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Ignoring")
            continue

        hands = hd.get_hands(frame, draw=True)
        fingers = hd.get_raised_fingers_angle()
        print(fingers)

        # Moving the mouse using the index finger
        if fingers == ["index"]:
            top_reduction = 100
            bottom_reduction = 200
            width_reduction = 150
            cv2.rectangle(frame, (width_reduction, top_reduction), (cam_width-width_reduction, cam_height-bottom_reduction),
                          (255, 0, 255), 3)
            location = hd.get_points([hd.INDEX_FINGER_TIP], draw=False, color=(255,0,255))
            if location:
                index_finger_tip = location[0]
                x1 = int(index_finger_tip["x"])
                y1 = int(index_finger_tip["y"])
                x = np.interp(x1, (width_reduction, cam_width-width_reduction), (0, scr_width))
                y = np.interp(y1, (top_reduction, cam_height-bottom_reduction), (0, scr_height))
                cloc_x = ploc_x + (x-ploc_x) / smooth
                cloc_y = ploc_y + (y-ploc_y) / smooth
                mouse.moveTo(cloc_x, cloc_y)
                ploc_x, ploc_y = cloc_x, cloc_y

        # Clicking
        cooldown = 2
        if fingers == ["index", "middle"]:
            distance = hd.get_distance([hd.INDEX_FINGER_TIP, hd.MIDDLE_FINGER_TIP])
            elapsed = time.time() - prev_click
            if distance < 30 and elapsed > cooldown:
                mouse.singleLeftClick()
                prev_click = time.time()

        # Scrolling
        if fingers == ['index', 'middle', 'ring', 'pinky']:
            mouse.scrollDown()

        if fingers == ['index', 'middle', 'ring',]:
            mouse.scrollUp()

        if fingers == ['thumb', 'pinky']:
            break

        # Volume control using index finger and thumb
        # locations = hd.get_points([hd.THUMB_TIP, hd.INDEX_FINGER_TIP], draw=True, color=(255,0,255))
        # if locations:
        #     thumb_tip_x, thumb_tip_y = locations[0]["x"], locations[0]["y"]
        #     index_tip_x, index_tip_y = locations[1]["x"], locations[1]["y"]
        #     length = math.hypot(thumb_tip_x-index_tip_x, thumb_tip_y-index_tip_y)
        #     print(length)

        

        current = time.time()
        fps = 1/(current-start)
        start = current

        # Showing the video from webcam
        output = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(output, str(int(fps)), (10, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow('Webcam', output)

        # Press q to quit
        if cv2.waitKey(1) == ord('q'):
            break

        

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
