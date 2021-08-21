import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode = False, max_num_hands = 1
                    , min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
mpDraw = mp.solutions.drawing_utils

start = 0
current = 0

while True:
    # Reading frame by frame
    ret, frame = cap.read()

    # Flipping the frame
    frame = cv2.flip(frame, 1)

    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Ignoring")
        continue

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
               h, w, c = frame.shape
               cx, cy = int(lm.x*w), int(lm.y*h)
               cv2.circle(frame, (cx, cy), id+10, (0, 255, 0), cv2.FILLED)
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    current = time.time()
    fps = 1/(current-start)
    start = current
    
    # Showing the video from webcam
    cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow('Webcam', frame)
    cv2.waitKey(1)

    # Press q to quit
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()