"""
MediaPipe 0.10.x Tasks API ile yüz takibi ve kafa açısı hesaplama.
transformation_matrixes KULLANILMIYOR — sadece solvePnP.
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


# ─── Model ────────────────────────────────────────────────────────────────────

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


# ─── Sabitler ─────────────────────────────────────────────────────────────────

# Gerçek insan yüzü 3D referans geometrisi (mm cinsinden)
FACE_3D_POINTS = np.array([
    (  0.0,    0.0,   0.0),    # Burun ucu       — landmark #4
    (  0.0, -330.0, -65.0),    # Çene             — landmark #152
    (-225.0,  170.0,-135.0),   # Sol göz dışı     — landmark #263
    ( 225.0,  170.0,-135.0),   # Sağ göz dışı     — landmark #33
    (-150.0, -150.0,-125.0),   # Sol ağız köşesi  — landmark #287
    ( 150.0, -150.0,-125.0),   # Sağ ağız köşesi  — landmark #57
], dtype=np.float64)

# Yukarıdaki 3D noktalara karşılık gelen landmark indeksleri
LANDMARK_IDS = [4, 152, 263, 33, 287, 57]


# ─── Veri Sınıfı ──────────────────────────────────────────────────────────────

@dataclass
class HeadPose:
    yaw:         float  # Sol(-) / Sağ(+)      derece
    pitch:       float  # Aşağı(-) / Yukarı(+) derece
    roll:        float  # Eğim                  derece
    landmarks:   list   # NormalizedLandmark listesi
    is_detected: bool


# ─── FaceTracker ──────────────────────────────────────────────────────────────

class FaceTracker:
    """
    MediaPipe 0.10.x Tasks API + solvePnP ile kafa açısı hesaplama.
    """

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
            # İkisi de False — sadece landmark koordinatları yeterli
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self._detector    = mp_vision.FaceLandmarker.create_from_options(opts)
        self._cam_matrix: Optional[np.ndarray] = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        print("[FaceTracker] Hazir (solvePnP modu)")

    # ── Kamera Matrisi ────────────────────────────────────────────────────────

    def _get_cam_matrix(self, w: int, h: int) -> np.ndarray:
        """
        Basit pinhole kamera matrisi.
        Gerçek kalibrasyon olmadan focal_length = frame genişliği
        iyi bir başlangıç yaklaşımıdır.
        """
        if self._cam_matrix is None:
            f = float(w)
            self._cam_matrix = np.array([
                [f,   0.,  w / 2.],
                [0.,  f,   h / 2.],
                [0.,  0.,  1.    ],
            ], dtype=np.float64)
        return self._cam_matrix

    # ── Rotasyon Matrisi → Euler Açıları ─────────────────────────────────────

    @staticmethod
    def _rot_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
        """
        3×3 rotasyon matrisini Euler açılarına çevirir.
        Gimbal-lock kontrolü dahildir.

        Returns:
            (yaw, pitch, roll) — derece cinsinden
        """
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        if sy > 1e-6:   # Normal durum
            roll  = np.arctan2( R[2, 1],  R[2, 2])
            pitch = np.arctan2(-R[2, 0],  sy)
            yaw   = np.arctan2( R[1, 0],  R[0, 0])
        else:            # Gimbal lock
            roll  = np.arctan2(-R[1, 2],  R[1, 1])
            pitch = np.arctan2(-R[2, 0],  sy)
            yaw   = 0.0

        return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

    # ── solvePnP ile Açı Hesaplama ────────────────────────────────────────────

    def _solve_pose(
        self,
        landmarks,
        w: int,
        h: int,
    ) -> Tuple[float, float, float]:
        """
        6 landmark noktası ile solvePnP çalıştırır,
        rotasyon matrisini Euler açılarına çevirir.

        Args:
            landmarks : NormalizedLandmark listesi
            w, h      : Kare boyutları (piksel)

        Returns:
            (yaw, pitch, roll) derece cinsinden; hata durumunda (0, 0, 0)
        """
        # Normalize koordinatları piksel koordinatına çevir
        pts_2d = np.array([
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in LANDMARK_IDS
        ], dtype=np.float64)

        cam_matrix = self._get_cam_matrix(w, h)

        success, rvec, _ = cv2.solvePnP(
            FACE_3D_POINTS,
            pts_2d,
            cam_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return 0.0, 0.0, 0.0

        # Rodrigues: rotasyon vektörü → 3×3 matris
        R, _ = cv2.Rodrigues(rvec)
        return self._rot_to_euler(R)

    # ── Ana Metot ─────────────────────────────────────────────────────────────

    def calculate_head_pose(self, frame: np.ndarray) -> HeadPose:
        """
        BGR kare alır, yüzü tespit eder ve kafa açısını döner.

        Args:
            frame: BGR formatında kamera karesi

        Returns:
            HeadPose nesnesi
        """
        h, w = frame.shape[:2]

        # BGR → RGB → MediaPipe Image
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Yüz landmark tespiti
        result = self._detector.detect(mp_img)

        # Yüz bulunamadı
        if not result.face_landmarks:
            return HeadPose(
                yaw=0.0, pitch=0.0, roll=0.0,
                landmarks=[], is_detected=False
            )

        landmarks = result.face_landmarks[0]  # İlk yüz

        # solvePnP ile açıları hesapla
        yaw, pitch, roll = self._solve_pose(landmarks, w, h)

        return HeadPose(
            yaw=yaw, pitch=pitch, roll=roll,
            landmarks=landmarks, is_detected=True
        )

    # ── Kontur Çizimi (Debug) ─────────────────────────────────────────────────

    def draw_face_contours(
        self,
        frame: np.ndarray,
        head_pose: HeadPose,
    ) -> None:
        """
        Landmark noktalarını frame üzerine çizer.

        - Küçük yeşil nokta : tüm landmarklar
        - Büyük turuncu daire: solvePnP'de kullanılan 6 ana nokta
        """
        if not head_pose.is_detected:
            return

        h, w = frame.shape[:2]

        # Tüm noktalar
        for lm in head_pose.landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (0, 200, 0), -1)

        # 6 referans noktası — vurgulu
        for idx in LANDMARK_IDS:
            lm = head_pose.landmarks[idx]
            x  = int(lm.x * w)
            y  = int(lm.y * h)
            cv2.circle(frame, (x, y), 5, (0, 80, 255), -1)

    # ── Temizlik ──────────────────────────────────────────────────────────────

    def release(self) -> None:
        """Landmarker nesnesini kapatır ve kaynakları serbest bırakır."""
        self._detector.close()
        print("[FaceTracker] Kaynaklar serbest birakildi.")