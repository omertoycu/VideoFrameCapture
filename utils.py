"""
Yardımcı fonksiyonlar: netlik kontrolü, görsel efektler, yüz kırpma.
Python 3.9+ ve MediaPipe 0.10.x uyumlu.
"""

import cv2
import numpy as np
from enum import Enum
from typing import Optional, Tuple


class CaptureAngle(Enum):
    FRONT = "on"
    LEFT  = "sol_profil"
    RIGHT = "sag_profil"


# ─── Netlik ───────────────────────────────────────────────────────────────────

def calculate_sharpness(frame: np.ndarray) -> float:
    """
    Laplacian varyansı ile netlik skorunu hesaplar.
    Yüksek değer → daha net görüntü.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_frame_sharp(
    frame: np.ndarray,
    threshold: float = 80.0
) -> Tuple[bool, float]:
    """
    Karenin net olup olmadığını kontrol eder.

    Returns:
        (is_sharp, sharpness_score)
    """
    score = calculate_sharpness(frame)
    return score >= threshold, score


# ─── Yüz Kırpma ───────────────────────────────────────────────────────────────

def crop_face_region(
    frame: np.ndarray,
    landmarks,
    padding: float = 0.3
) -> Optional[np.ndarray]:
    """
    Landmark listesinden yüz bölgesini kırpar ve padding ekler.
    None döner eğer kırpma başarısız olursa.
    """
    h, w = frame.shape[:2]

    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]

    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    pad_x = int((x_max - x_min) * padding)
    pad_y = int((y_max - y_min) * padding)

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(w, x_max + pad_x)
    y_max = min(h, y_max + pad_y)

    # Geçerli alan kontrolü
    if x_max <= x_min or y_max <= y_min:
        return None

    cropped = frame[y_min:y_max, x_min:x_max]

    # Boş array kontrolü — 'or' yerine explicit None dön
    if cropped is None or cropped.size == 0:
        return None

    return cropped


# ─── Görsel Efektler ──────────────────────────────────────────────────────────

def draw_flash_effect(frame: np.ndarray, alpha: float) -> np.ndarray:
    """
    Yeşil kenarlık flaş efekti uygular.

    Args:
        frame : BGR kare
        alpha : Yoğunluk (0.0 → 1.0)

    Returns:
        Efekt uygulanmış kare (yeni kopya)
    """
    if alpha <= 0:
        return frame

    out = frame.copy()
    thickness = max(8, int(frame.shape[0] * 0.025))
    overlay   = out.copy()

    cv2.rectangle(
        overlay, (0, 0),
        (frame.shape[1], frame.shape[0]),
        (0, 255, 0), thickness * 2
    )
    return cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)


def draw_angle_indicator(
    frame: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
    position: Tuple[int, int] = (10, 100)
) -> None:
    """Açı bilgilerini ekrana yazar."""
    x, y  = position
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.60
    thick = 2

    lines = [
        (f"Yaw   (Sol/Sag) : {yaw:+.1f} deg",   (0, 200, 255)),
        (f"Pitch (Yukari/As): {pitch:+.1f} deg", (0, 200, 255)),
        (f"Roll  (Egim)     : {roll:+.1f} deg",  (0, 200, 255)),
    ]

    for i, (text, color) in enumerate(lines):
        cy = y + i * 26
        # Gölge
        cv2.putText(frame, text, (x + 1, cy + 1),
                    font, scale, (0, 0, 0), thick + 2)
        # Ana metin
        cv2.putText(frame, text, (x, cy),
                    font, scale, color, thick)


def draw_capture_status(
    frame: np.ndarray,
    captured_angles: dict,
    position: Optional[Tuple[int, int]] = None
) -> None:
    """
    Sol/Ön/Sağ yakalama durumunu ekranın sol alt köşesine çizer.
    """
    h, w  = frame.shape[:2]
    x, y  = position or (10, h - 50)
    font  = cv2.FONT_HERSHEY_SIMPLEX

    items = [
        (CaptureAngle.LEFT,  "<-- Sol"),
        (CaptureAngle.FRONT, "  On  "),
        (CaptureAngle.RIGHT, "Sag -->"),
    ]

    for i, (angle, label) in enumerate(items):
        done  = captured_angles.get(angle, False)
        color = (0, 255, 0) if done else (120, 120, 120)
        icon  = "[OK]" if done else "[ ]"
        cv2.putText(frame, f"{icon} {label}",
                    (x + i * 160, y),
                    font, 0.65, (0, 0, 0), 4)
        cv2.putText(frame, f"{icon} {label}",
                    (x + i * 160, y),
                    font, 0.65, color, 2)


def draw_sharpness_bar(
    frame: np.ndarray,
    sharpness: float,
    threshold: float
) -> None:
    """Sağ üst köşede netlik göstergesi çizer."""
    h, w = frame.shape[:2]

    bx, by  = w - 160, 70
    bw, bh_ = 140, 14
    max_val = 300.0

    norm      = min(sharpness / max_val, 1.0)
    fill      = int(bw * norm)
    thr_x     = bx + int(bw * (threshold / max_val))
    bar_color = (0, 220, 0) if sharpness >= threshold else (0, 80, 220)

    # Arka plan
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh_), (40, 40, 40), -1)
    # Dolgu
    if fill > 0:
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bh_), bar_color, -1)
    # Eşik çizgisi
    cv2.line(frame, (thr_x, by - 3), (thr_x, by + bh_ + 3), (255, 220, 0), 2)
    # Kenarlık
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh_), (180, 180, 180), 1)
    # Etiket
    cv2.putText(frame, f"Netlik: {sharpness:.0f}",
                (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (200, 200, 200), 1)


def draw_guide_overlay(
    frame: np.ndarray,
    angle_type: Optional[CaptureAngle]
) -> None:
    """Üst orta şerit: kullanıcıya yön rehberi."""
    h, w = frame.shape[:2]

    if angle_type is None:
        text, color = "Kafanizi cevirin: On / Sol / Sag", (180, 180, 180)
    elif angle_type == CaptureAngle.FRONT:
        text, color = "On: Duz bakin", (0, 255, 255)
    elif angle_type == CaptureAngle.LEFT:
        text, color = "<-- Sola donun", (0, 165, 255)
    else:
        text, color = "Saga donun -->", (0, 165, 255)

    size   = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    tx     = (w - size[0]) // 2
    ty     = 40

    cv2.rectangle(frame, (tx - 10, ty - 26), (tx + size[0] + 10, ty + 8),
                  (0, 0, 0), -1)
    cv2.rectangle(frame, (tx - 10, ty - 26), (tx + size[0] + 10, ty + 8),
                  color, 2)
    cv2.putText(frame, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def draw_completion_screen(frame: np.ndarray) -> None:
    """Tüm açılar yakalandığında tamamlama ekranı."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 150, 0), -1)
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

    msgs = [
        ("TUM ACLAR YAKALANDI!", 1.1, (0, 255, 80),    3),
        ("Fotograflar diske kaydedildi.",  0.7, (230, 230, 230), 2),
        ("[q] Cikis   [r] Yeniden Baslat", 0.62, (180, 180, 180), 1),
    ]
    base_y = h // 2 - 55

    for i, (msg, sc, col, th) in enumerate(msgs):
        sz  = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, sc, th)[0]
        tx  = (w - sz[0]) // 2
        ty  = base_y + i * 60
        cv2.putText(frame, msg, (tx + 2, ty + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 2)
        cv2.putText(frame, msg, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, col, th)