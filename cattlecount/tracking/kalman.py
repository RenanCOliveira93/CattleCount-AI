"""Filtro de Kalman de velocidade constante no espaço (cx, cy, a, h) — a = largura/altura — como no ByteTrack/DeepSORT."""
from __future__ import annotations

import numpy as np


class KalmanFilterXYAH:
    def __init__(self, std_pos: float = 1 / 20, std_vel: float = 1 / 160):
        self.std_pos, self.std_vel = std_pos, std_vel
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, 4 + i] = 1.0
        self.H = np.eye(4, 8)

    def initiate(self, z: np.ndarray):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self.std_pos * h, 2 * self.std_pos * h, 1e-2, 2 * self.std_pos * h,
               10 * self.std_vel * h, 10 * self.std_vel * h, 1e-5, 10 * self.std_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        q = np.diag(np.square([self.std_pos * h, self.std_pos * h, 1e-2, self.std_pos * h,
                               self.std_vel * h, self.std_vel * h, 1e-5, self.std_vel * h]))
        return self.F @ mean, self.F @ cov @ self.F.T + q

    def update(self, mean, cov, z):
        h = mean[3]
        r = np.diag(np.square([self.std_pos * h, self.std_pos * h, 1e-1, self.std_pos * h]))
        S = self.H @ cov @ self.H.T + r
        K = np.linalg.solve(S.T, (cov @ self.H.T).T).T
        mean = mean + K @ (z - self.H @ mean)
        cov = cov - K @ S @ K.T
        return mean, cov


def xyxy_to_xyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h])


def xyah_to_xyxy(z):
    w = z[2] * z[3]
    return np.array([z[0] - w / 2, z[1] - z[3] / 2, z[0] + w / 2, z[1] + z[3] / 2])
