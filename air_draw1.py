import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opening")
    exit()
else:
    print("Camera connected successfully")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
draw = mp.solutions.drawing_utils

canvas = None
prev_x, prev_y = 0, 0

print("Air Draw Started!")

while True:
    success, frame = cap.read()

    if not success:
        print("Camera error")
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:

            draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            h, w, c = frame.shape

            x = int(hand.landmark[8].x * w)
            y = int(hand.landmark[8].y * h)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            cv2.line(
                canvas,
                (prev_x, prev_y),
                (x, y),
                (0, 255, 0),
                5
            )

            prev_x, prev_y = x, y

    else:
        prev_x, prev_y = 0, 0

    frame = cv2.add(frame, canvas)

    cv2.imshow("My Air Draw Project", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()     