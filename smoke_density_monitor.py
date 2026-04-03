#!/usr/bin/env python3
"""
Smoke Density Estimation and Air Quality Monitoring using Dark Channel Prior (DCP).

Implements the atmospheric scattering model:
    I(x) = J(x) * t(x) + A * (1 - t(x))

This script estimates smoke density from images/video/webcam streams by:
1) Computing dark channel prior
2) Estimating atmospheric light
3) Estimating and refining transmission map
4) Computing smoke density = 1 - mean(transmission)
5) Classifying air quality category
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


# --------------------------- Core DCP Functions --------------------------- #

def dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    Compute dark channel for an RGB/BGR image.

    Steps:
      1) Per-pixel min over color channels
      2) Minimum filter using erosion with square patch

    Args:
        image: Input BGR image in uint8 [0,255] or float [0,1].
        patch_size: Odd patch size for local minimum filter.

    Returns:
        Dark channel map in float32 [0,1].
    """
    if patch_size < 1:
        raise ValueError("patch_size must be >= 1")

    if patch_size % 2 == 0:
        patch_size += 1

    image_f = image.astype(np.float32)
    if image_f.max() > 1.0:
        image_f /= 255.0

    min_rgb = np.min(image_f, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_rgb, kernel)
    return dark


def estimate_atmospheric_light(
    image: np.ndarray,
    dark_channel_map: np.ndarray,
    top_percent: float = 0.001,
) -> np.ndarray:
    """
    Estimate atmospheric light A.

    Procedure:
      1) Select top 0.1% brightest pixels in dark channel
      2) Among those locations, choose the highest intensity pixel from original image

    Args:
        image: Input BGR image.
        dark_channel_map: Dark channel in [0,1].
        top_percent: Fraction of brightest dark channel pixels to use (0.001 = 0.1%).

    Returns:
        Atmospheric light vector A as float32 array of shape (3,) in [0,1].
    """
    image_f = image.astype(np.float32)
    if image_f.max() > 1.0:
        image_f /= 255.0

    h, w = dark_channel_map.shape
    num_pixels = h * w
    num_top = max(int(num_pixels * top_percent), 1)

    dark_vec = dark_channel_map.reshape(-1)
    image_vec = image_f.reshape(-1, 3)

    top_indices = np.argpartition(dark_vec, -num_top)[-num_top:]
    candidate_pixels = image_vec[top_indices]
    candidate_intensity = np.sum(candidate_pixels, axis=1)

    best_idx = np.argmax(candidate_intensity)
    A = candidate_pixels[best_idx]
    A = np.clip(A, 1e-6, 1.0)
    return A.astype(np.float32)


def estimate_transmission(
    image: np.ndarray,
    atmospheric_light: np.ndarray,
    omega: float = 0.95,
    patch_size: int = 15,
) -> np.ndarray:
    """
    Estimate coarse transmission map:
        t(x) = 1 - omega * dark_channel(I(x)/A)

    Args:
        image: Input BGR image.
        atmospheric_light: A vector shape (3,) in [0,1].
        omega: DCP retention factor; higher means stronger haze/smoke removal.
        patch_size: Patch size for dark channel.

    Returns:
        Coarse transmission map in float32 [0,1].
    """
    image_f = image.astype(np.float32)
    if image_f.max() > 1.0:
        image_f /= 255.0

    normalized = image_f / atmospheric_light.reshape(1, 1, 3)
    dark_norm = dark_channel(normalized, patch_size=patch_size)
    transmission = 1.0 - omega * dark_norm
    transmission = np.clip(transmission, 0.0, 1.0)
    return transmission


def refine_transmission(
    transmission: np.ndarray,
    image: Optional[np.ndarray] = None,
    radius: int = 40,
    eps: float = 1e-3,
    fallback_kernel: int = 9,
) -> np.ndarray:
    """
    Refine transmission map using guided filter when available; otherwise Gaussian blur.

    Args:
        transmission: Coarse transmission in [0,1].
        image: Guidance BGR image.
        radius: Guided filter radius.
        eps: Guided filter epsilon.
        fallback_kernel: Gaussian kernel size fallback.

    Returns:
        Refined transmission map in [0,1].
    """
    transmission_f = transmission.astype(np.float32)

    if image is not None and hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        guide = image.astype(np.float32)
        if guide.max() > 1.0:
            guide /= 255.0
        guide_gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
        refined = cv2.ximgproc.guidedFilter(
            guide=guide_gray,
            src=transmission_f,
            radius=radius,
            eps=eps,
            dDepth=-1,
        )
    else:
        if fallback_kernel % 2 == 0:
            fallback_kernel += 1
        refined = cv2.GaussianBlur(transmission_f, (fallback_kernel, fallback_kernel), 0)

    return np.clip(refined, 0.0, 1.0)


def compute_smoke_density(transmission: np.ndarray) -> float:
    """Compute smoke density as 1 - mean(transmission)."""
    density = 1.0 - float(np.mean(transmission))
    return float(np.clip(density, 0.0, 1.0))


def classify_air_quality(density: float) -> str:
    """Classify air quality from smoke density."""
    if density <= 0.2:
        return "Good"
    if density <= 0.5:
        return "Moderate"
    if density <= 0.7:
        return "Unhealthy"
    return "Hazardous"


# ---------------------------- Visualization ------------------------------ #

def normalize_to_uint8(gray: np.ndarray) -> np.ndarray:
    gray = np.clip(gray, 0.0, 1.0)
    return (gray * 255.0).astype(np.uint8)


def make_transmission_histogram(transmission: np.ndarray, width: int = 512, height: int = 220) -> np.ndarray:
    """Render transmission histogram as an image for dashboard display."""
    hist, _ = np.histogram(transmission.flatten(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    hist /= max(hist.max(), 1.0)

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    for x in range(1, 256):
        x1 = int((x - 1) * (width - 1) / 255)
        x2 = int(x * (width - 1) / 255)
        y1 = int((1.0 - hist[x - 1]) * (height - 25))
        y2 = int((1.0 - hist[x]) * (height - 25))
        cv2.line(canvas, (x1, y1), (x2, y2), (50, 120, 220), 2)

    cv2.putText(canvas, "Transmission Histogram", (12, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    return canvas


def add_overlay(frame: np.ndarray, density: float, label: str, fps: Optional[float] = None) -> np.ndarray:
    """Overlay smoke metrics on frame."""
    out = frame.copy()
    pct = density * 100.0

    if label == "Good":
        color = (60, 200, 60)
    elif label == "Moderate":
        color = (0, 210, 255)
    elif label == "Unhealthy":
        color = (0, 140, 255)
    else:
        color = (0, 0, 255)

    cv2.rectangle(out, (8, 8), (500, 110), (0, 0, 0), -1)
    cv2.putText(out, f"Smoke Density: {pct:.2f}%", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(out, f"Air Quality: {label}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    if fps is not None:
        cv2.putText(out, f"FPS: {fps:.2f}", (320, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 2)

    return out


def build_dashboard(
    original: np.ndarray,
    dark_map: np.ndarray,
    transmission: np.ndarray,
    heatmap: np.ndarray,
    hist_image: np.ndarray,
    density: float,
    label: str,
    fps: Optional[float] = None,
) -> np.ndarray:
    """Create a multi-panel dashboard for display/save."""
    h, w = original.shape[:2]

    dark_u8 = normalize_to_uint8(dark_map)
    dark_bgr = cv2.cvtColor(dark_u8, cv2.COLOR_GRAY2BGR)

    trans_u8 = normalize_to_uint8(transmission)
    trans_bgr = cv2.cvtColor(trans_u8, cv2.COLOR_GRAY2BGR)

    panel_original = add_overlay(original, density, label, fps)

    panel_dark = dark_bgr
    cv2.putText(panel_dark, "Dark Channel", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    panel_trans = trans_bgr
    cv2.putText(panel_trans, "Transmission Map", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    panel_heat = heatmap.copy()
    cv2.putText(panel_heat, "Smoke Heatmap", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    hist_resized = cv2.resize(hist_image, (w, h), interpolation=cv2.INTER_AREA)

    top_row = np.hstack([panel_original, panel_dark])
    mid_row = np.hstack([panel_trans, panel_heat])
    bottom_row = np.hstack([hist_resized, np.full_like(hist_resized, 245)])

    cv2.putText(bottom_row, f"Smoke Density: {density:.4f} | Air Quality: {label}",
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)

    dashboard = np.vstack([top_row, mid_row, bottom_row])
    return dashboard


# ---------------------------- Processing Core ---------------------------- #

def process_frame(frame: np.ndarray, patch_size: int, omega: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """Process single frame/image and return intermediates + metrics."""
    dark_map = dark_channel(frame, patch_size=patch_size)
    A = estimate_atmospheric_light(frame, dark_map)
    transmission = estimate_transmission(frame, A, omega=omega, patch_size=patch_size)
    transmission_refined = refine_transmission(transmission, image=frame)

    density = compute_smoke_density(transmission_refined)
    label = classify_air_quality(density)

    smoke_strength = normalize_to_uint8(1.0 - transmission_refined)
    heatmap = cv2.applyColorMap(smoke_strength, cv2.COLORMAP_JET)

    return dark_map, transmission_refined, heatmap, density, label


def maybe_resize(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0 or scale > 1.0:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def process_image(args: argparse.Namespace) -> None:
    image_path = Path(args.source)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    image = maybe_resize(image, args.process_scale)

    dark_map, transmission, heatmap, density, label = process_frame(image, args.patch_size, args.omega)
    hist_img = make_transmission_histogram(transmission)
    dashboard = build_dashboard(image, dark_map, transmission, heatmap, hist_img, density, label, fps=None)

    if args.save_output:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), dashboard)
        print(f"[INFO] Saved output image: {output_path}")

    print(f"[RESULT] Smoke density: {density:.4f} ({density * 100.0:.2f}%)")
    print(f"[RESULT] Air quality: {label}")

    if not args.no_display:
        cv2.imshow("Smoke Density Monitoring (Image)", dashboard)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _parse_video_source(source: str) -> Union[int, str]:
    if source.isdigit():
        return int(source)
    return source


def process_video(args: argparse.Namespace) -> None:
    source = _parse_video_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.source}")

    writer = None
    if args.save_output:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if args.process_scale > 0 and args.process_scale <= 1.0:
            frame_w = int(frame_w * args.process_scale)
            frame_h = int(frame_h * args.process_scale)

        dashboard_w, dashboard_h = frame_w * 2, frame_h * 3
        fps_out = cap.get(cv2.CAP_PROP_FPS)
        if fps_out <= 0:
            fps_out = 20.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps_out, (dashboard_w, dashboard_h))

    prev_t = time.time()
    smoothed_fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = maybe_resize(frame, args.process_scale)

        dark_map, transmission, heatmap, density, label = process_frame(frame, args.patch_size, args.omega)
        hist_img = make_transmission_histogram(transmission)

        curr_t = time.time()
        instant_fps = 1.0 / max(curr_t - prev_t, 1e-6)
        prev_t = curr_t
        smoothed_fps = instant_fps if smoothed_fps == 0.0 else 0.9 * smoothed_fps + 0.1 * instant_fps

        dashboard = build_dashboard(frame, dark_map, transmission, heatmap, hist_img, density, label, fps=smoothed_fps)

        if writer is not None:
            writer.write(dashboard)

        if not args.no_display:
            cv2.imshow("Smoke Density Monitoring (Video/Webcam)", dashboard)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[INFO] Saved output video: {args.output_path}")
    cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke Density Estimation and Air Quality Monitoring using Dark Channel Prior (DCP)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["image", "video", "webcam"],
        required=True,
        help="Input mode: image, video, or webcam",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Path to image/video file or webcam index (e.g., 0)",
    )

    parser.add_argument("--patch-size", type=int, default=15, help="Patch size for dark channel minimum filter")
    parser.add_argument("--omega", type=float, default=0.95, help="Omega parameter for transmission estimation")
    parser.add_argument("--t0", type=float, default=0.1, help="Reserved lower bound for transmission (future extension)")
    parser.add_argument(
        "--process-scale",
        type=float,
        default=1.0,
        help="Scale factor in (0,1] for faster processing; 1.0 means original size",
    )

    parser.add_argument("--save-output", action="store_true", help="Save dashboard output to file")
    parser.add_argument("--output-path", type=str, default="outputs/smoke_monitor_output.mp4", help="Output path")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV visualization windows")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "image":
        if args.source == "0":
            raise ValueError("For image mode, provide --source <image_path>")
        if args.output_path.endswith(".mp4") and args.save_output:
            args.output_path = "outputs/smoke_monitor_output.png"
        process_image(args)
        return

    if args.mode == "webcam" and args.source == "0":
        args.source = "0"

    process_video(args)


if __name__ == "__main__":
    main()
