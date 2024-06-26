import cv2
import mediapipe as mp

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert the frame color from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hand landmarks
    results = hands.process(frame_rgb)
    
    total_fingers = 0  # To store the total number of fingers
    
    # Draw hand landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmark positions
            landmark_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([cx, cy])
                
            # Finger counting logic
            if len(landmark_list) != 0:
                fingers = []
                
                # Thumb: Check if thumb tip is to the right of thumb MCP (for right hand)
                if landmark_list[4][0] > landmark_list[3][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                
                # Other four fingers
                for tip in [8, 12, 16, 20]:
                    if landmark_list[tip][1] < landmark_list[tip - 2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                
                total_fingers += fingers.count(1)
    
    # Display the total number of fingers on the frame
    cv2.putText(frame, f'Total Fingers: {total_fingers}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
