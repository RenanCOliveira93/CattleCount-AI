"""Compensação de movimento global de câmera (drone): estima uma transformação afim parcial (translação+rotação+escala)
entre quadros consecutivos por fluxo óptico esparso (LK) + RANSAC. Sem isso, a associação por IoU falha quando o drone se move
mais rápido que o animal. Se a estimativa falhar, devolve identidade (e conta a falha)."""
from __future__ import annotations

import cv2
import numpy as np


class GlobalMotion:
    def __init__(self, max_corners: int = 400, downscale: int = 2):
        self.prev = None
        self.failures = 0
        self.max_corners, self.downscale = max_corners, downscale

    def update(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Retorna A (2×3) tal que p_atual ≈ A @ [p_anterior; 1] (coordenadas do quadro completo)."""
        g = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        s = 1.0 / self.downscale
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        A = np.array([[1, 0, 0], [0, 1, 0]], float)
        if self.prev is not None:
            p0 = cv2.goodFeaturesToTrack(self.prev, self.max_corners, 0.01, 8)
            if p0 is not None and len(p0) >= 8:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, p0, None)
                ok = st.ravel() == 1
                if ok.sum() >= 8:
                    M, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    if M is not None and inl is not None and inl.sum() >= 6:
                        A = M.astype(float); A[:, 2] /= s
                    else:
                        self.failures += 1
                else:
                    self.failures += 1
            else:
                self.failures += 1
        self.prev = g
        return A


def apply_affine(A: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Aplica A a caixas (n,4) como transformação de seus 4 cantos, devolvendo a caixa envolvente."""
    b = np.asarray(xyxy, float).reshape(-1, 4)
    pts = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]], 1)
    q = pts @ A[:, :2].T + A[:, 2]
    return np.c_[q[..., 0].min(1), q[..., 1].min(1), q[..., 0].max(1), q[..., 1].max(1)]
