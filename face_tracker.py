"""
MediaPipe 0.10.x Tasks API ile yüz takibi ve kafa açısı hesaplama.
Eksen yeniden ataması düzeltildi.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import urllib.request


MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = "face_landmarker.task"


def download_model_if_needed(path: str = MODEL_PATH) -> str:
    if Path(path).exists():
        print(f"[Model] Mevcut: {path}")
        return path
    print(f"[Model] Indiriliyor (~29 MB)...")

    def _hook(blk, blk_sz, total):
        pct = min(blk * blk_sz / total * 100, 100) if total else 0
        bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
        print(f"\r  [{bar}] {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(MODEL_URL, path, _hook)
    print(f"\n[Model] Indirildi: {path}")
    return path


# ─── 3D Referans Noktaları ────────────────────────────────────────────────────
# Standart yüz geometrisi (mm) — OpenCV koordinat sistemi
FACE_3D_POINTS = np.array([
    [  0.0,    0.0,    0.0  ],   # Burun ucu        #4
    [  0.0,  -63.6,  -12.5 ],   # Çene              #152
    [-43.3,   32.7,  -26.0 ],   # Sol göz dış köşe  #263
    [ 43.3,   32.7,  -26.0 ],   # Sağ göz dış köşe  #33
    [-28.9,  -28.9,  -24.1 ],   # Sol ağız köşesi   #287
    [ 28.9,  -28.9,  -24.1 ],   # Sağ ağız köşesi   #57
], dtype=np.float64)

LANDMARK_IDS = [4, 152, 263, 33, 287, 57]


@dataclass
class HeadPose:
    yaw:         float   # Sol(-) / Sağ(+)      derece
    pitch:       float   # Yukarı(+) / Aşağı(-) derece
    roll:        float   # Eğim                  derece
    landmarks:   list
    is_detected: bool


class FaceTracker:

    def __init__(
        self,
        model_path:           str   = MODEL_PATH,
        detection_confidence: float = 0.70,
        tracking_confidence:  float = 0.70,
        num_faces:            int   = 1,
    ):
        model_path = download_model_if_needed(model_path)

        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=num_faces,
            min_face_detection_confidence=detection_confidence,
            min_face_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self._detector    = mp_vision.FaceLandmarker.create_from_options(opts)
        self._cam_matrix: Optional[np.ndarray] = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        print("[FaceTracker] Hazir (solvePnP + eksen duzeltmesi)")

    def _get_cam_matrix(self, w: int, h: int) -> np.ndarray:
        if self._cam_matrix is None:
            f = float(w)
            self._cam_matrix = np.array([
                [f,   0.,  w / 2.],
                [0.,  f,   h / 2.],
                [0.,  0.,  1.    ],
            ], dtype=np.float64)
        return self._cam_matrix

    def _solve_pose(
        self,
        landmarks,
        w: int,
        h: int,
    ) -> Tuple[float, float, float]:
        """
        solvePnP ile rotasyon vektörü hesaplar,
        ardından yüz koordinat sistemine göre
        Yaw/Pitch/Roll'u doğru eksenlere atar.
        """
        pts_2d = np.array([
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in LANDMARK_IDS
        ], dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            FACE_3D_POINTS,
            pts_2d,
            self._get_cam_matrix(w, h),
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return 0.0, 0.0, 0.0

        # Rotasyon vektörü → matris
        R, _ = cv2.Rodrigues(rvec)

        # ── Euler açılarını çıkar ─────────────────────────────────────────
        # solvePnP kamera koordinat sisteminde çalışır:
        #   X ekseni: sağa
        #   Y ekseni: aşağıya
        #   Z ekseni: kameradan dışarıya
        #
        # Yüz açıları için:
        #   Yaw   = Y ekseni etrafında dönüş (sol/sağ)
        #   Pitch = X ekseni etrafında dönüş (yukarı/aşağı)
        #   Roll  = Z ekseni etrafında dönüş (eğim)

        # Pitch (X ekseni) — aşağı/yukarı
        pitch = np.degrees(np.arcsin(-R[1, 2]))

        # Yaw (Y ekseni) — sol/sağ
        yaw = np.degrees(np.arctan2(R[0, 2], R[2, 2]))

        # Roll (Z ekseni) — eğim
        roll = np.degrees(np.arctan2(-R[1, 0], R[1, 1]))

        return yaw, pitch, roll

    def calculate_head_pose(self, frame: np.ndarray) -> HeadPose:
        h, w = frame.shape[:2]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_img)

        if not result.face_landmarks:
            return HeadPose(0., 0., 0., [], False)

        landmarks = result.face_landmarks[0]
        yaw, pitch, roll = self._solve_pose(landmarks, w, h)

        return HeadPose(yaw, pitch, roll, landmarks, True)

    def draw_face_contours(self, frame: np.ndarray, head_pose: HeadPose) -> None:
        if not head_pose.is_detected:
            return
        h, w = frame.shape[:2]

        for lm in head_pose.landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 200, 0), -1)

        for idx in LANDMARK_IDS:
            lm = head_pose.landmarks[idx]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 80, 255), -1)

    def release(self) -> None:
        self._detector.close()
        print("[FaceTracker] Kaynaklar serbest birakildi.")