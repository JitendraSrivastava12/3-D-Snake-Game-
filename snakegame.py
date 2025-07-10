import cv2
import numpy as np
import math
import cvzone
import random
import streamlit as st
import av
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from cvzone.HandTrackingModule import HandDetector

# ---------- Snake Game Class ----------
class SnakeGame:
    def __init__(self, path_food, food_count=3):
        self.points = []
        self.length = []
        self.current_length = 0
        self.prev_snake_head = (0, 0)
        self.allowed_length = 50
        self.game_over = False
        self.food = cv2.imread(path_food, cv2.IMREAD_UNCHANGED)
        self.food = cv2.resize(self.food, (50, 50))
        self.hfood, self.wfood, _ = self.food.shape
        self.food_count = food_count
        self.food_points = []
        self.food_location()
        self.score = 0

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

        return img


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Snake Game", layout="centered")

if "game" not in st.session_state:
    st.session_state["game"] = SnakeGame("apple_00.png", food_count=3)
if "top_score" not in st.session_state:
    st.session_state["top_score"] = 0

st.title("🐍 Snake Game - Hand Gesture Controlled")
st.markdown("Use your **index finger** 🖐️ to control the snake. Eat 🍎 and avoid crashing!")

if st.button("🔄 Restart Game"):
    st.session_state["top_score"] = max(st.session_state["top_score"], st.session_state["game"].score)
    st.session_state["game"] = SnakeGame("apple_00.png", food_count=3)


# ---------- Video Processor Class ----------
class GameProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = HandDetector(detectionCon=0.5, maxHands=1)
        self.prev_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        curr_time = time.time()
        run_game = (curr_time - self.prev_time) > 0.1

        try:
            if "game" not in st.session_state:
                st.session_state["game"] = SnakeGame("apple_00.png", food_count=3)

            if run_game:
                self.prev_time = curr_time
                hands, _ = self.detector.findHands(img, draw=False)
                if hands:
                    lmList = hands[0]["lmList"]
                    x, y = lmList[8][0], lmList[8][1]
                    img = st.session_state["game"].update((x, y), img)
                else:
                    cvzone.putTextRect(img, "Show Your Hand!", [300, 300], scale=2, thickness=2, offset=8)
                    img = st.session_state["game"].update(st.session_state["game"].prev_snake_head, img)

                score = st.session_state["game"].score
                st.session_state["top_score"] = max(st.session_state["top_score"], score)

                cvzone.putTextRect(img, f"Score: {score}", [50, 30], scale=2, thickness=2, offset=5)
                cvzone.putTextRect(img, f"Top: {st.session_state['top_score']}", [900, 30], scale=2, thickness=2, offset=5)

                if st.session_state["game"].game_over:
                    cvzone.putTextRect(img, "Game Over", [400, 250], scale=3, thickness=3, offset=10, colorT=(255, 255, 255), colorR=(255, 0, 0))

        except Exception as e:
            print("[ERROR in recv()]", e)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------- WebRTC Camera ----------
webrtc_streamer(
    key="snake-game",
    video_processor_factory=GameProcessor,  # ✅ updated for latest version
    media_stream_constraints={
        "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
        "audio": False
    },
    async_processing=True
)
