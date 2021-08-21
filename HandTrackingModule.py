import cv2
import mediapipe as mp
import time
import numpy as np
import math 


class Hand_Detector():
    """The 21 hand landmarks."""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def __init__(self, mode=False, num_hands=1,
                 detection_confidence=0.7, tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.engine = self.mp_hands.Hands(
            mode, num_hands, detection_confidence, tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        self.hands_landmarks = None
        self.image = None

    def get_hands(self, img, draw=False):
        self.image = img
        self.hands_landmarks = self.engine.process(img).multi_hand_landmarks
        if self.hands_landmarks and draw==True:
            for handLms in self.hands_landmarks:
                self.mp_draw.draw_landmarks(self.image, handLms, self.mp_hands.HAND_CONNECTIONS)

        return self.hands_landmarks

    def get_points(self, list_of_index, draw=False, color=(255, 0, 0)):
        height, width, channel = self.image.shape
        list_of_points = []
        if self.hands_landmarks:
            for index in list_of_index:
                landmarks_list = self.hands_landmarks[0].landmark
                point = {
                    "x" : int(landmarks_list[index].x*width),
                    "y" : int(landmarks_list[index].y*height),
                    "z" : landmarks_list[index].z
                }
                list_of_points.append(point)
                if draw == True:
                    cv2.circle(self.img, (point["x"], point["y"]), 5, color, cv2.FILLED)
            return list_of_points
        else:
            return None

    def get_distance(self, list_of_points):
        height, width, channel = self.image.shape
        distance = None
        if self.hands_landmarks:
            distance = 0
            landmarks_list = self.hands_landmarks[0].landmark
            for index in range(len(list_of_points)-1):
                # x1, y1, z1 = landmarks_list[index].x, landmarks_list[index].y, landmarks_list[index].z
                # x2, y2, z2 = landmarks_list[index+1].x, landmarks_list[index+1].y, landmarks_list[index+1].z
                # distance = distance + math.hypot((x1-x2), (y1-y2), (z1-z2))
                x1, y1= int(landmarks_list[list_of_points[index]].x*width), int(landmarks_list[list_of_points[index]].y*height)
                x2, y2= int(landmarks_list[list_of_points[index+1]].x*width), int(landmarks_list[list_of_points[index+1]].y*height)
                cv2.line(self.image, (x1,y1), (x2,y2), (255,0,0), 3)
                distance = distance + math.hypot((x1-x2), (y1-y2))
        return distance

    def get_angle(self, p1, p2, p3):
        angle = None
        if self.hands_landmarks:
            landmarks_list = self.hands_landmarks[0].landmark
            a = np.array([landmarks_list[p1].x, landmarks_list[p1].y])
            b = np.array([landmarks_list[p2].x, landmarks_list[p2].y])
            c = np.array([landmarks_list[p3].x, landmarks_list[p3].y])
            ba = a-b
            bc = c-b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            return np.degrees(angle)

    def get_raised_fingers_angle(self):
        fingers = []
        if self.hands_landmarks:
            landmarks_list = self.hands_landmarks[0].landmark
            if self.get_angle(2, 3, 4) > 140 and landmarks_list[4].x > landmarks_list[2].x:
                fingers.append("thumb")
            if self.get_angle(5, 6, 7) > 120:
                fingers.append("index")
            if self.get_angle(9, 10, 11) > 120:
                fingers.append("middle")
            if self.get_angle(13, 14, 15) > 120:
                fingers.append("ring")
            if self.get_angle(17, 18, 19) > 120:
                fingers.append("pinky")
        return fingers

    def get_raised_fingers(self):
        fingers = []
        if self.hands_landmarks:
            landmarks_list = self.hands_landmarks[0].landmark
            if landmarks_list[4].x > landmarks_list[2].x:
                fingers.append("thumb")
            if landmarks_list[8].y < landmarks_list[6].y:
                fingers.append("index")
            if landmarks_list[12].y < landmarks_list[10].y:
                fingers.append("middle")
            if landmarks_list[16].y < landmarks_list[14].y:
                fingers.append("ring")
            if landmarks_list[20].y < landmarks_list[18].y:
                fingers.append("pinky")
        return fingers


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    start = 0
    current = 0

    hd = Hand_Detector()
    
    while True:
        # Reading frame by frame
        ret, frame = cap.read()

        # Flipping the frame
        frame = cv2.flip(frame, 1)

        # If frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Ignoring")
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hands = hd.get_hands(img_rgb)
        point = hd.get_point(img_rgb, hd.INDEX_FINGER_TIP)
        if point:
            print(point["x"])
        else:
            print("None")
        img_rgb = hd.draw_hands(img_rgb)
        img_rgb = hd.draw_points(img_rgb, [hd.INDEX_FINGER_TIP, hd.MIDDLE_FINGER_TIP])

        current = time.time()
        fps = 1/(current-start)
        start = current

        # Showing the video from webcam
        output = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(output, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow('Webcam', output)

        # Press q to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
