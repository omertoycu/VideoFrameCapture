"""
Ana uygulama döngüsü.
"""

import cv2
import numpy as np
import argparse
import sys
import time
from typing import Optional

from face_tracker    import FaceTracker
from capture_manager import CaptureManager
from utils import (
    CaptureAngle,
    draw_flash_effect,
    draw_angle_indicator,
    draw_capture_status,
    draw_sharpness_bar,
    draw_guide_overlay,
    draw_completion_screen,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Yüz Açısı Otomatik Çekim POC")
    p.add_argument("--camera",          type=int,   default=0)
    p.add_argument("--width",           type=int,   default=1280)
    p.add_argument("--height",          type=int,   default=720)
    p.add_argument("--threshold",       type=float, default=80.0)
    p.add_argument("--output",          type=str,   default="captured_faces")
    p.add_argument("--no-mesh",         action="store_true")
    p.add_argument("--save-full-frame", action="store_true")
    return p.parse_args()


def open_camera(idx: int, w: int, h: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"[HATA] Kamera {idx} açılamadı.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    rw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rf = cap.get(cv2.CAP_PROP_FPS)
    print(f"[Kamera] {rw}x{rh} @ {rf:.0f} fps")
    return cap


def main() -> None:
    args = parse_args()

    print("=" * 52)
    print("   Yüz Açısı Tabanlı Otomatik Çekim POC")
    print("=" * 52)
    print(f"  Kamera        : {args.camera}")
    print(f"  Netlik Esigi  : {args.threshold}")
    print(f"  Kayit Klasoru : {args.output}")
    print("=" * 52)
    print("  [q/ESC] Cikis  [r] Sifirla  [s] Ekran goruntusu  [m] Mesh")
    print("=" * 52)

    cap             = open_camera(args.camera, args.width, args.height)
    tracker         = FaceTracker()
    capture_manager = CaptureManager(
        output_dir          = args.output,
        sharpness_threshold = args.threshold,
        save_full_frame     = args.save_full_frame,
    )

    show_mesh       = not args.no_mesh
    fps_time        = time.time()
    fps_count       = 0
    current_fps     = 0.0
    last_sharpness  = 0.0      # UI için saklanır

    print("\n[✓] Hazir. Yüzünüzü kameraya gösterin.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[HATA] Kamera karesi alinamadi.")
            break

        frame       = cv2.flip(frame, 1)
        clean_frame = frame.copy()          # çizimsiz kopya

        # ── Yüz Tespiti & Açı Hesaplama ─────────────────────────────────────
        head_pose = tracker.calculate_head_pose(frame)

        # ── Otomatik Çekim ───────────────────────────────────────────────────
        cap_result = {
            "angle_type": None,
            "sharpness":  last_sharpness,
            "captured":   False,
        }

        if head_pose.is_detected:
            cap_result = capture_manager.process_frame(
                clean_frame = clean_frame,
                yaw         = head_pose.yaw,
                pitch       = head_pose.pitch,
                roll        = head_pose.roll,
                landmarks   = head_pose.landmarks,
            )

        # Netlik değerini her zaman güncelle (yüz algılansa da algılanmasa da)
        # process_frame artık her zaman sharpness döndürüyor
        if cap_result["sharpness"] > 0:
            last_sharpness = cap_result["sharpness"]

        # ── UI Katmanları ────────────────────────────────────────────────────

        # 1. Mesh / kontur noktaları
        if show_mesh and head_pose.is_detected:
            tracker.draw_face_contours(frame, head_pose)

        # 2. Açı bilgisi (sol üst)
        if head_pose.is_detected:
            draw_angle_indicator(
                frame,
                head_pose.yaw,
                head_pose.pitch,
                head_pose.roll,
            )

        # 3. Netlik çubuğu (sağ üst) — last_sharpness kullan
        draw_sharpness_bar(frame, last_sharpness, args.threshold)

        # 4. Yön rehberi (üst orta)
        if not capture_manager.all_captured:
            draw_guide_overlay(frame, cap_result.get("angle_type"))

        # 5. Yakalama durum şeridi (alt)
        draw_capture_status(frame, capture_manager.captured_angles)

        # 6. Flaş efekti
        alpha = capture_manager.max_flash_alpha
        if alpha > 0:
            frame = draw_flash_effect(frame, alpha)

        # 7. Tamamlama ekranı
        if capture_manager.all_captured:
            draw_completion_screen(frame)

        # 8. Yüz algılanmadı uyarısı
        if not head_pose.is_detected:
            h, w  = frame.shape[:2]
            msg   = "Yuz algilanamadi"
            sz    = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.putText(
                frame, msg,
                ((w - sz[0]) // 2, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 220), 2
            )

        # 9. FPS sayacı
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            current_fps = fps_count / (time.time() - fps_time)
            fps_count   = 0
            fps_time    = time.time()

        cv2.putText(
            frame, f"FPS:{current_fps:.0f}",
            (frame.shape[1] - 90, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1
        )

        cv2.imshow("Cilt Analizi POC", frame)

        # ── Klavye Kontrolleri ───────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):           # Çıkış
            print("\n[!] Cikis yapildi.")
            break
        elif key == ord('r'):               # Sıfırla
            capture_manager.reset()
            last_sharpness = 0.0
            print("[r] Yeni oturum basladi.")
        elif key == ord('s'):               # Ekran görüntüsü
            fn = f"screenshot_{time.strftime('%H%M%S')}.jpg"
            cv2.imwrite(fn, frame)
            print(f"[s] Kaydedildi: {fn}")
        elif key == ord('m'):               # Mesh aç/kapat
            show_mesh = not show_mesh
            print(f"[m] Mesh: {'acik' if show_mesh else 'kapali'}")

        # ── DEBUG: Açıları terminale yaz ────────────────────────
        if head_pose.is_detected:
            print(
                f"\rYaw:{head_pose.yaw:+6.1f} "
                f"Pitch:{head_pose.pitch:+6.1f} "
                f"Roll:{head_pose.roll:+6.1f}  ",
                end="", flush=True
            )

    cap.release()
    tracker.release()
    cv2.destroyAllWindows()
    print("[✓] Uygulama kapatildi.")


if __name__ == "__main__":
    main()