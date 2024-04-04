import cv2
import mediapipe as mp
import pyautogui
import time
from collections import deque

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

mp_drawing = mp.solutions.drawing_utils

left_hand_gesture = None
right_hand_gesture = None

gesture_queue = deque(maxlen=10)

last_slide_time = time.time()
slide_delay = 1  # Adjust the delay between slide changes (in seconds)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_landmark_list = hand_landmarks.landmark

            hand_x = hand_landmark_list[mp_hands.HandLandmark.WRIST].x
            thumb_y = hand_landmark_list[mp_hands.HandLandmark.THUMB_TIP].y
            index_finger_y = hand_landmark_list[
                mp_hands.HandLandmark.INDEX_FINGER_TIP
            ].y

            if hand_x < 0.5:
                if index_finger_y < thumb_y:
                    left_hand_gesture = "pointing_up"
                elif index_finger_y > thumb_y:
                    left_hand_gesture = "pointing_down"
            else:
                if index_finger_y < thumb_y:
                    right_hand_gesture = "pointing_up"
                elif index_finger_y > thumb_y:
                    right_hand_gesture = "pointing_down"

    if left_hand_gesture == "pointing_up":
        current_time = time.time()
        if current_time - last_slide_time > slide_delay:
            pyautogui.hotkey("alt", "f5")  # Open slideshow
            print("Open Slideshow")
            last_slide_time = current_time
    elif left_hand_gesture == "pointing_down":
        current_time = time.time()
        if current_time - last_slide_time > slide_delay:
            pyautogui.press("esc")  # Close slideshow
            print("Close Slideshow")
            last_slide_time = current_time

    if right_hand_gesture == "pointing_up":
        gesture_queue.append("pointing_up")
    elif right_hand_gesture == "pointing_down":
        gesture_queue.append("pointing_down")

    if "pointing_up" in gesture_queue:
        current_time = time.time()
        if current_time - last_slide_time > slide_delay:
            pyautogui.press("right")  # Next slide
            print("Next Slide")
            last_slide_time = current_time
        gesture_queue.clear()
    elif "pointing_down" in gesture_queue:
        current_time = time.time()
        if current_time - last_slide_time > slide_delay:
            pyautogui.press("left")  # Previous slide
            print("Previous Slide")
            last_slide_time = current_time
        gesture_queue.clear()

    left_hand_gesture = None
    right_hand_gesture = None

    if results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
