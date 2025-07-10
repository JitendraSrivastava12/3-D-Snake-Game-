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
        if self.game_over:
            return img  # Do not update anything after game over

        px, py = self.prev_snake_head
        cx, cy = current_head
        cx = int(0.7 * px + 0.3 * cx)  # Smooth X
        cy = int(0.7 * py + 0.3 * cy)  # Smooth Y

        dist = math.hypot(px - cx, py - cy)
        self.length.append(dist)
        self.current_length += dist
        self.prev_snake_head = cx, cy
        self.points.append([cx, cy])

        while self.current_length > self.allowed_length and self.length:
            self.current_length -= self.length[0]
            self.length.pop(0)
            self.points.pop(0)

        # Draw snake line
        for i in range(1, len(self.points)):
            cv2.line(img, tuple(self.points[i - 1]), tuple(self.points[i]), (0, 255, 0), 6)

        if self.points:
            cv2.circle(img, tuple(self.points[-1]), 10, (0, 0, 255), -1)  # Head

        # Collision detection
        if len(self.points) >= 10:
            pts = np.array(self.points[:-5], np.int32).reshape(-1, 1, 2)
            mdist = cv2.pointPolygonTest(pts, (cx, cy), True)
            if mdist > -5 and mdist < 5:
                self.game_over = True
                return img

        # Eat food
        for i, (rx, ry) in enumerate(self.food_points):
            if rx - self.wfood // 2 < cx < rx + self.wfood // 2 and ry - self.hfood // 2 < cy < ry + self.hfood // 2:
                self.food_points[i] = (random.randint(100, 1000), random.randint(100, 500))
                self.allowed_length += 40
                self.score += 1

        # Draw food
        for rx, ry in self.food_points:
            try:
                if self.food.shape[2] == 4:
                    overlay_img = self.food[:, :, :3]
                    mask = self.food[:, :, 3] > 0
                    for c in range(3):
                        img[ry]()
