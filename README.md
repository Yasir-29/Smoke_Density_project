# Smoke Density Estimation and Air Quality Monitoring (Dark Channel Prior)

A complete OpenCV + NumPy project for estimating smoke density from **images**, **video files**, or **webcam streams** using the **Dark Channel Prior (DCP)** method and atmospheric scattering model.

## Model

The implementation follows:

\[
I(x) = J(x) \cdot t(x) + A \cdot (1 - t(x))
\]

Where:
- `I(x)` = observed image
- `J(x)` = scene radiance
- `t(x)` = transmission map
- `A` = atmospheric light

## Features

- Dark Channel Prior pipeline:
  - RGB min channel
  - 15x15 (default) local minimum filter
  - Atmospheric light from top 0.1% brightest dark-channel pixels
  - Transmission estimation with `omega` parameter
  - Guided filter refinement (if `cv2.ximgproc` is available), Gaussian fallback otherwise
- Smoke density estimation:
  - `smoke_density = 1 - mean(transmission)`
- Air quality classification:
  - Good: `0.0 - 0.2`
  - Moderate: `0.2 - 0.5`
  - Unhealthy: `0.5 - 0.7`
  - Hazardous: `> 0.7`
- Dashboard visualization includes:
  - Original frame + overlays
  - Dark channel
  - Transmission map
  - Smoke heatmap
  - Transmission histogram
- Real-time webcam processing
- FPS display and processing scale optimization
- Save output image/video dashboard

## Project Files

- `smoke_density_monitor.py` – full implementation
- `requirements.txt` – dependencies

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Usage

### 1) Image input

```bash
python smoke_density_monitor.py \
  --mode image \
  --source path/to/image.jpg \
  --patch-size 15 \
  --omega 0.95 \
  --save-output \
  --output-path outputs/result.png
```

### 2) Video file input

```bash
python smoke_density_monitor.py \
  --mode video \
  --source path/to/video.mp4 \
  --process-scale 0.75 \
  --save-output \
  --output-path outputs/result.mp4
```

### 3) Webcam input

```bash
python smoke_density_monitor.py \
  --mode webcam \
  --source 0 \
  --process-scale 0.7
```

## CLI Arguments

- `--mode {image,video,webcam}` (required)
- `--source` input path or camera index (default: `0`)
- `--patch-size` DCP patch size (default: `15`)
- `--omega` transmission parameter (default: `0.95`)
- `--process-scale` speed optimization scale in `(0,1]` (default: `1.0`)
- `--save-output` save dashboard output
- `--output-path` output image/video path
- `--no-display` disable OpenCV windows

## Notes

- For guided filter refinement, install `opencv-contrib-python` instead of `opencv-python` to get `cv2.ximgproc`:

```bash
pip uninstall -y opencv-python
pip install opencv-contrib-python
```

- Press `q` or `Esc` to stop video/webcam processing.

