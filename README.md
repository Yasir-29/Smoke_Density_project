# Smoke Density Estimation and Air Quality Monitoring using Dark Channel Prior (DCP)

This project estimates **smoke/haze density from images** (no physical sensors) using the **Dark Channel Prior (DCP)** algorithm and then classifies **air quality** into:

- **Good**
- **Moderate**
- **Hazardous**

Core idea:

- **Transmission** \(t\) drops as smoke/haze increases
- **Smoke Density** \(= 1 - \text{Transmission}\)

## Dataset (RESIDE ITS)

Place the dataset in `dataset/` with:

- `dataset/haze/` (input hazy images)
- `dataset/trans/` (ground-truth transmission maps; 1=clear, 0=dense)
- `dataset/clear/` (clean reference images; not required for training here)

Filename pairing expected (typical RESIDE ITS):

- haze: `10001_01_0.9797.png`
- trans: `10001_01.png`

## Pipeline

- Load paired samples from `haze/` and `trans/`
- Preprocess: resize to **256×256**, normalize to **0–1**
- Apply DCP on haze image:
  - `dark_channel(image)`
  - `atmospheric_light(image, dark)`
  - `transmission(image, A)`
- Extract features from estimated transmission map:
  - mean, std, min, max
- Generate labels from **ground-truth transmission**:
  - Good: density < 0.2
  - Moderate: density < 0.5
  - Hazardous: otherwise
- Train `RandomForestClassifier`
- Evaluate with accuracy, classification report, confusion matrix
- Predict for a new image and display:
  - original haze image
  - estimated transmission map
  - smoke density %
  - predicted air quality label

## Installation

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Train

```bash
py main.py train --dataset_dir dataset --limit -1 --cv_folds 5 --model_out outputs/model.joblib
```

Use fewer samples (faster):

```bash
py main.py train --dataset_dir dataset --limit 500 --cv_folds 5
```

## Predict (single image)

```bash
python main.py predict --model outputs/model.joblib --image dataset/haze/your_image.png
```

## Project Files

- `preprocessing.py`: dataset loading + preprocessing
- `feature_extraction.py`: DCP functions + feature extraction + label logic
- `main.py`: training/evaluation + prediction/visualization CLI

