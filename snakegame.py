import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import cvzone
import numpy as np
import random
import vlc

class SnakeGame:
    def __init__(self, path_food, food_count=3):  # Allow multiple food items
        self.points = []
        self.length = []
        self.current_length = 0
        self.prev_snake_head = 0, 0
        self.allowed_length = 50
        self.game_over = False
        self.food = cv2.imread(path_food, cv2.IMREAD_UNCHANGED)
        self.food = cv2.resize(self.food, (50, 60))
        self.hfood, self.wfood, _ = self.food.shape
        self.food_count = food_count
        self.food_points = []
        self.food_location()
        self.score = 0

    def food_location(self):
        """Generate multiple food positions"""
        self.food_points = [(random.randint(100, 600), random.randint(100, 400)) for _ in range(self.food_count)]

    def update(self, current_head, img):
        global player, player1

        if self.game_over:
            cvzone.putTextRect(img, 'Game Over', [160, 150], scale=3, thickness=3, offset=20)
            if player.is_playing():
                player.stop()
            if not player1.is_playing():
                player1.play()
        else:
            if not player.is_playing():
                player.play()

            px, py = self.prev_snake_head
            cx, cy = current_head
            dist = math.hypot(px - cx, py - cy)
            self.length.append(dist)
            self.current_length += dist
            self.prev_snake_head = cx, cy
            self.points.append([cx, cy])

            while self.current_length > self.allowed_length and self.length:
                self.current_length -= self.length[0]
                self.length.pop(0)
                self.points.pop(0)

            # Check collision with each food item
            for food_index, (rx, ry) in enumerate(self.food_points):
                if rx - self.wfood // 2 < cx < rx + self.wfood // 2 and ry - self.hfood // 2 < cy < ry + self.hfood // 2:
                    self.food_points[food_index] = (random.randint(100, 600), random.randint(100, 400))  # Replace eaten food
                    self.allowed_length += 40
                    self.score += 1

            if len(self.points) > 1:
                for i in range(1, len(self.points)):
                    cv2.circle(img, tuple(self.points[i - 1]), 10, (255, 0, 0), -1)

            if self.points:
                cv2.circle(img, tuple(self.points[-1]), 10, (0, 0, 255), -1)

            if len(self.points) >= 3:
                pts = np.array(self.points[:-2], np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], False, (0, 255, 0), 6)
                mdist = cv2.pointPolygonTest(pts, (cx, cy), True)
                if -1 <= mdist <= 1:
                    self.game_over = True

            # Draw multiple food items
            for rx, ry in self.food_points:
                img = cvzone.overlayPNG(img, self.food, (rx - self.wfood // 2, ry - self.hfood // 2))

            cvzone.putTextRect(img, f"Score: {self.score}", [70, 30], scale=3, thickness=3, offset=7)

        return img


# --- Main Game Loop ---
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.5, maxHands=1)

# Media players
player = vlc.MediaPlayer("/home/pi/opencvsnakegame/m.mp3")
player1 = vlc.MediaPlayer("/home/pi/opencvsnakegame/m1.mp3")

gameplay = SnakeGame('apple_00.png', food_count=3)  # Set multiple food items

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame, exiting...")
        break

    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img)
    if hands:
        lmList = hands[0]['lmList']
        x, y = lmList[8][0], lmList[8][1]
        img = gameplay.update((x, y), img)

    cv2.imshow("Snake Game", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        gameplay = SnakeGame('apple_00.png', food_count=3)  # Reset the game with multiple food items

cap.release()
cv2.destroyAllWindows()