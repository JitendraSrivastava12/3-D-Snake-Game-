import cv2
import numpy as np
import math
import random
import streamlit as st
import av
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import mediapipe as mp

# ---------- Snake Game ----------
class SnakeGame:
    def __init__(self, path_food="apple_00.png", food_count=3):
        self.points = []
        self.length = []
        self.current_length = 0
        self.prev_snake_head = (0, 0)
        self.allowed_length = 50
        self.game_over = False
        self.score = 0
        self.top_score = 0
        self.food_count = food_count

        self.food = cv2.imread(path_food, cv2.IMREAD_UNCHANGED)
        if self.food is None:
            print("⚠️ apple_00.png not found. Using green box as food.")
            self.food = np.zeros((50, 50, 4), dtype=np.uint8)
            self.food[:, :, 1] = 255
            self.hfood, self.wfood = 50, 50
        else:
            self.food = cv2.resize(self.food, (50, 50))
            self.hfood, self.wfood, _ = self.food.shape

        self.food_points = []
        self.food_location()

    def food_location(self):
        self.food_points = [(random.randint(100, 1000), random.randint(100, 500)) for _ in range(self.food_count)]

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
                self.food_points[food_index] = (random.randint(100, 1000), random.randint(100, 500))
                self.allowed_length += 40
                self.score += 1

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
            try:
                # fallback draw if PNG fails
                if self.food.shape[2] == 4:
                    overlay_img = self.food[:, :, :3]
                    mask = self.food[:, :, 3] > 0
                    for c in range(3):
                        img[ry - 25:ry + 25, rx - 25:rx + 25, c][mask] = overlay_img[:, :, c][mask]
                else:
                    cv2.rectangle(img, (rx - 25, ry - 25), (rx + 25, ry + 25), (0, 255, 0), -1)
            except:
                cv2.rectangle(img, (rx - 20, ry - 20), (rx + 20, ry + 20), (0, 255, 0), -1)

        return img


# ---------- Raw MediaPipe Hand Tracking (No cvzone) ----------
class GameProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,         # ✅ CPU only (no GPU)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.draw = mp.solutions.drawing_utils
        self.game = SnakeGame()
        self.last_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        try:
            current_time = time.time()
            if current_time - self.last_time > 0.1:
                self.last_time = current_time

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    h, w, _ = img.shape
                    x = int(hand.landmark[8].x * w)
                    y = int(hand.landmark[8].y * h)
                    img = self.game.update((x, y), img)
                else:
                    img = self.game.update(self.game.prev_snake_head, img)
                    cv2.putText(img, "Show Your Hand!", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                self.game.top_score = max(self.game.top_score, self.game.score)
                cv2.putText(img, f"Score: {self.game.score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(img, f"Top: {self.game.top_score}", (900, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)

                if self.game.game_over:
                    cv2.putText(img, "Game Over", (400, 250), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)

        except Exception as e:
            print("❌ recv error:", e)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Snake Game", layout="centered")
st.title("🐍 Snake Game - Hand Gesture Controlled")
st.markdown("Use your **index finger** 🖐️ to move the snake. Eat 🍎 and avoid crashing!")

webrtc_streamer(
    key="snake-game",
    video_processor_factory=GameProcessor,
    media_stream_constraints={
        "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
        "audio": False
    },
    async_processing=True
)
