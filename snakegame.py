import cv2
import numpy as np
import math
import cvzone
import random
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from cvzone.HandTrackingModule import HandDetector

# Game logic class
class SnakeGame:
    def __init__(self, path_food, food_count=3):
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
        self.food_points = [(random.randint(100, 600), random.randint(100, 400)) for _ in range(self.food_count)]

    def update(self, current_head, img):
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

        for food_index, (rx, ry) in enumerate(self.food_points):
            if rx - self.wfood // 2 < cx < rx + self.wfood // 2 and ry - self.hfood // 2 < cy < ry + self.hfood // 2:
                self.food_points[food_index] = (random.randint(100, 600), random.randint(100, 400))
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

        for rx, ry in self.food_points:
            img = cvzone.overlayPNG(img, self.food, (rx - self.wfood // 2, ry - self.hfood // 2))

        text = "Game Over" if self.game_over else f"Score: {self.score}"
        cvzone.putTextRect(img, text, [70, 30], scale=3, thickness=3, offset=7)

        return img

# Streamlit video transformer
class GameTransformer(VideoTransformerBase):
    def __init__(self):
        self.game = SnakeGame('apple_00.png', food_count=3)
        self.detector = HandDetector(detectionCon=0.5, maxHands=1)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        hands, img = self.detector.findHands(img, draw=False)
        if hands:
            lmList = hands[0]["lmList"]
            x, y = lmList[8][0], lmList[8][1]
            img = self.game.update((x, y), img)
        return img

# UI
st.title("🐍 Snake Game with Hand Tracking")
st.markdown("Use your index finger to control the snake. Pinch food to eat!")

webrtc_streamer(
    key="snake-game",
    video_transformer_factory=GameTransformer,
    media_stream_constraints={
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
        "audio": False,
    },
    async_processing=True,
)
