"""
Çekim yönetimi: açı doğrulama, netlik kontrolü,
temiz kare kaydetme, flaş efekti zamanlayıcısı.
"""

import cv2
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from utils import CaptureAngle, is_frame_sharp, crop_face_region


# ─── Yapılandırma ─────────────────────────────────────────────────────────────

@dataclass
class AngleConfig:
    """Bir çekim açısının kabul aralığı."""
    angle_type:      CaptureAngle
    yaw_min:         float
    yaw_max:         float
    pitch_tolerance: float = 18.0
    roll_tolerance:  float = 18.0


@dataclass
class CaptureState:
    """Tek bir açı için çekim + animasyon durumu."""
    is_captured:      bool  = False
    flash_start:      float = 0.0
    flash_duration:   float = 0.55   # saniye
    last_capture:     float = 0.0
    cooldown:         float = 2.0    # saniye

    # ── Flash alpha (0→1→0 sinüs eğrisi) ─────────────────────────────────────
    @property
    def flash_alpha(self) -> float:
        if self.flash_start == 0.0:
            return 0.0
        elapsed = time.time() - self.flash_start
        if elapsed >= self.flash_duration:
            return 0.0
        return float(np.sin(elapsed / self.flash_duration * np.pi) * 0.65)

    @property
    def is_flashing(self) -> bool:
        return self.flash_alpha > 0.0

    def trigger_flash(self) -> None:
        self.flash_start  = time.time()
        self.last_capture = time.time()

    def in_cooldown(self) -> bool:
        return (time.time() - self.last_capture) < self.cooldown


# ─── CaptureManager ───────────────────────────────────────────────────────────

class CaptureManager:
    """
    Her kare için:
      1. Yaw/pitch/roll açılarının hedef aralıkta olup olmadığını kontrol eder.
      2. Laplacian ile netliği ölçer.
      3. Koşullar sağlandığında temiz kareyi (çizimsiz) diske yazar.
      4. Flaş animasyonu için zamanlayıcı tutar.
    """

    ANGLE_CONFIGS = [
        AngleConfig(CaptureAngle.FRONT, yaw_min=-15.0, yaw_max=15.0),
        AngleConfig(CaptureAngle.LEFT,  yaw_min=-60.0, yaw_max=-35.0),
        AngleConfig(CaptureAngle.RIGHT, yaw_min= 35.0, yaw_max= 60.0),
    ]

    def __init__(
        self,
        output_dir:          str   = "captured_faces",
        sharpness_threshold: float = 80.0,
        save_full_frame:     bool  = False,
        session_id:          Optional[str] = None,
    ):
        self.sharpness_threshold = sharpness_threshold
        self.save_full_frame     = save_full_frame
        self.session_id          = session_id or time.strftime("%Y%m%d_%H%M%S")

        self.session_dir = Path(output_dir) / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        print(f"[CaptureManager] Kayit klasoru: {self.session_dir}")

        self.states: dict[CaptureAngle, CaptureState] = {
            a: CaptureState() for a in CaptureAngle
        }

    # ── İç Yardımcılar ───────────────────────────────────────────────────────

    def _active_config(
        self, yaw: float, pitch: float, roll: float
    ) -> Optional[AngleConfig]:
        for cfg in self.ANGLE_CONFIGS:
            if (cfg.yaw_min <= yaw <= cfg.yaw_max
                    and abs(pitch) <= cfg.pitch_tolerance
                    and abs(roll)  <= cfg.roll_tolerance):
                return cfg
        return None

    def _save(
            self,
            clean: np.ndarray,
            landmarks,
            angle: CaptureAngle,
    ) -> Optional[str]:
        ts = time.strftime("%H%M%S")
        path = self.session_dir / f"{angle.value}_{ts}.jpg"

        # Kayıt için kullanılacak görüntüyü belirle
        if self.save_full_frame or not landmarks:
            img = clean
        else:
            cropped = crop_face_region(clean, landmarks, 0.3)
            # ← 'or' YOK, explicit None kontrolü var
            if cropped is not None:
                img = cropped
            else:
                img = clean

        ok = cv2.imwrite(str(path), img)
        if ok:
            print(f"[OK] {angle.value} kaydedildi -> {path}")
            return str(path)
        print(f"[!!] Kayit basarisiz: {path}")
        return None

    # ── Ana Metot ─────────────────────────────────────────────────────────────

    def process_frame(
        self,
        clean_frame: np.ndarray,
        yaw: float,
        pitch: float,
        roll: float,
        landmarks,
    ) -> dict:
        """
        Her döngü adımında çağrılır.

        Args:
            clean_frame : Üzerinde çizim OLMAYAN orijinal kare (kayıt için).
            yaw/pitch/roll : Hesaplanmış açılar.
            landmarks   : NormalizedLandmark listesi (yüz kırpma için).

        Returns:
            {
              'captured'   : bool,
              'angle_type' : CaptureAngle | None,
              'sharpness'  : float,
              'filepath'   : str | None,
            }
        """
        result = dict(captured=False, angle_type=None,
                      sharpness=0.0, filepath=None)

        cfg = self._active_config(yaw, pitch, roll)
        if cfg is None:
            return result

        result['angle_type'] = cfg.angle_type
        state = self.states[cfg.angle_type]

        if state.is_captured or state.in_cooldown():
            return result

        sharp_ok, score = is_frame_sharp(clean_frame, self.sharpness_threshold)
        result['sharpness'] = score

        if not sharp_ok:
            return result

        fp = self._save(clean_frame, landmarks, cfg.angle_type)
        if fp:
            state.is_captured = True
            state.trigger_flash()
            result.update(captured=True, filepath=fp)

        return result

    # ── Özellikler ────────────────────────────────────────────────────────────

    @property
    def captured_angles(self) -> dict:
        return {a: s.is_captured for a, s in self.states.items()}

    @property
    def all_captured(self) -> bool:
        return all(s.is_captured for s in self.states.values())

    @property
    def max_flash_alpha(self) -> float:
        return max(s.flash_alpha for s in self.states.values())

    def reset(self) -> None:
        for s in self.states.values():
            s.is_captured  = False
            s.flash_start  = 0.0
            s.last_capture = 0.0
        self.session_id  = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.session_dir.parent / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        print(f"[CaptureManager] Sifirlandi. Yeni klasor: {self.session_dir}")