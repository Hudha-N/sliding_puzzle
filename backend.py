import cv2
import mediapipe as mp
import numpy as np
import random
import time

# ---------------------------
# MediaPipe Setup
# ---------------------------
mp_hands = mp.tasks.vision
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp_hands.HandLandmarker
HandLandmarkerOptions = mp_hands.HandLandmarkerOptions
VisionRunningMode = mp_hands.RunningMode
ImageFormat = mp.ImageFormat

model_path = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# ---------------------------
# Gesture Variables
# ---------------------------
prev_x, prev_y = None, None
last_gesture_time = 0
gesture_cooldown = 0.4

# ---------------------------
# Puzzle Setup
# ---------------------------
GRID_SIZE = 3
TILE_SIZE = 100
WINDOW_MARGIN = 50
GRID_COLOR = (255, 255, 255)
TILE_COLOR = (0, 128, 255)
FONT_COLOR = (255, 255, 255)

# Create solved puzzle
puzzle = [[GRID_SIZE * r + c + 1 for c in range(GRID_SIZE)] for r in range(GRID_SIZE)]
puzzle[GRID_SIZE-1][GRID_SIZE-1] = 0  # empty tile

# Shuffle puzzle
def shuffle_puzzle(puzzle, moves=50):
    empty_r, empty_c = GRID_SIZE-1, GRID_SIZE-1
    for _ in range(moves):
        direction = random.choice(['up','down','left','right'])
        if direction == 'up' and empty_r < GRID_SIZE-1:
            puzzle[empty_r][empty_c], puzzle[empty_r+1][empty_c] = puzzle[empty_r+1][empty_c], puzzle[empty_r][empty_c]
            empty_r += 1
        elif direction == 'down' and empty_r > 0:
            puzzle[empty_r][empty_c], puzzle[empty_r-1][empty_c] = puzzle[empty_r-1][empty_c], puzzle[empty_r][empty_c]
            empty_r -= 1
        elif direction == 'left' and empty_c < GRID_SIZE-1:
            puzzle[empty_r][empty_c], puzzle[empty_r][empty_c+1] = puzzle[empty_r][empty_c+1], puzzle[empty_r][empty_c]
            empty_c += 1
        elif direction == 'right' and empty_c > 0:
            puzzle[empty_r][empty_c], puzzle[empty_r][empty_c-1] = puzzle[empty_r][empty_c-1], puzzle[empty_r][empty_c]
            empty_c -= 1
shuffle_puzzle(puzzle)

# Find empty tile
def find_empty(puzzle):
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if puzzle[r][c] == 0:
                return r, c
    return None, None

# ---------------------------
# Draw Puzzle
# ---------------------------
def draw_puzzle(puzzle):
    canvas_size = TILE_SIZE*GRID_SIZE + 2*WINDOW_MARGIN
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            tile = puzzle[r][c]
            top_left = (WINDOW_MARGIN + c*TILE_SIZE, WINDOW_MARGIN + r*TILE_SIZE)
            bottom_right = (top_left[0]+TILE_SIZE, top_left[1]+TILE_SIZE)
            if tile != 0:
                cv2.rectangle(canvas, top_left, bottom_right, TILE_COLOR, -1)
                cv2.putText(canvas, str(tile),
                            (top_left[0]+TILE_SIZE//3, top_left[1]+TILE_SIZE//2+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, FONT_COLOR, 2)
            else:
                cv2.rectangle(canvas, top_left, bottom_right, (50,50,50), -1)
            cv2.rectangle(canvas, top_left, bottom_right, GRID_COLOR, 2)
    return canvas

# ---------------------------
# Camera
# ---------------------------
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        gesture_text = ""

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]

            wrist = hand_landmarks[0]
            current_x = wrist.x
            current_y = wrist.y
            current_time = time.time()

            if prev_x is not None and prev_y is not None and current_time - last_gesture_time > gesture_cooldown:
                diff_x = current_x - prev_x
                diff_y = current_y - prev_y
                empty_r, empty_c = find_empty(puzzle)

                # Horizontal move
                if abs(diff_x) > abs(diff_y):
                    if diff_x > 0.07 and empty_c > 0:  # Swipe Right → move tile left into empty
                        puzzle[empty_r][empty_c], puzzle[empty_r][empty_c-1] = puzzle[empty_r][empty_c-1], puzzle[empty_r][empty_c]
                        gesture_text = "Swipe Right"
                        last_gesture_time = current_time
                    elif diff_x < -0.07 and empty_c < GRID_SIZE-1:  # Swipe Left → move tile right into empty
                        puzzle[empty_r][empty_c], puzzle[empty_r][empty_c+1] = puzzle[empty_r][empty_c+1], puzzle[empty_r][empty_c]
                        gesture_text = "Swipe Left"
                        last_gesture_time = current_time

                # Vertical move
                else:
                    if diff_y > 0.07 and empty_r > 0:  # Swipe Down → move tile up into empty
                        puzzle[empty_r][empty_c], puzzle[empty_r-1][empty_c] = puzzle[empty_r-1][empty_c], puzzle[empty_r][empty_c]
                        gesture_text = "Swipe Down"
                        last_gesture_time = current_time
                    elif diff_y < -0.07 and empty_r < GRID_SIZE-1:  # Swipe Up → move tile down into empty
                        puzzle[empty_r][empty_c], puzzle[empty_r+1][empty_c] = puzzle[empty_r+1][empty_c], puzzle[empty_r][empty_c]
                        gesture_text = "Swipe Up"
                        last_gesture_time = current_time

            prev_x = current_x
            prev_y = current_y

        # Draw puzzle
        game_canvas = draw_puzzle(puzzle)

        # Show gesture text
        if gesture_text:
            cv2.putText(game_canvas, f"Gesture: {gesture_text}",
                        (50, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("Puzzle Game", game_canvas)
        cv2.imshow("Camera", frame)

        # Check for win
        solved = all(puzzle[r][c] == GRID_SIZE*r + c +1 for r in range(GRID_SIZE) for c in range(GRID_SIZE-1)) and puzzle[GRID_SIZE-1][GRID_SIZE-1]==0
        if solved:
            cv2.putText(game_canvas, "You Win!", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
            cv2.imshow("Puzzle Game", game_canvas)
            cv2.waitKey(3000)
            break

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
