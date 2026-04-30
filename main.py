from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from feature_extraction import extract_features, generate_label, label_to_name
from preprocessing import load_dataset_with_ids, preprocess_image


def _build_xy(haze_images: np.ndarray, trans_images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X: list[list[float]] = []
    y: list[int] = []

    for haze, trans_gt in zip(haze_images, trans_images, strict=True):
        feats, _t_est = extract_features(haze)
        X.append([float(v) for v in feats])
        y.append(int(generate_label(trans_gt)))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train_model(
    dataset_dir: Path,
    limit: int | None,
    test_size: float,
    random_state: int,
    n_estimators: int,
    cv_folds: int,
    model_out: Path,
) -> None:
    haze_dir = dataset_dir / "haze"
    trans_dir = dataset_dir / "trans"

    haze_images, trans_images, scene_ids = load_dataset_with_ids(
        str(haze_dir), str(trans_dir), limit=limit, size=(256, 256)
    )
    X, y = _build_xy(haze_images, trans_images)
    unique_scenes = len(np.unique(scene_ids))
    class_counts = np.bincount(y, minlength=3)
    print(f"Loaded samples: {len(X)}")
    print(f"Unique scenes (group ids): {unique_scenes}")
    print(f"Class counts [Low, Moderate, High]: {class_counts.tolist()}")

    # Critical for RESIDE: split by *scene* to prevent leakage
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=scene_ids))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    scene_ids_train = scene_ids[train_idx]

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt",
    )
    clf.fit(X_train, y_train)

    if cv_folds and cv_folds > 1:
        # Group-aware CV inside the training split for a less noisy estimate.
        gkf = GroupKFold(n_splits=cv_folds)
        cv_scores: list[float] = []
        for tr, va in gkf.split(X_train, y_train, groups=scene_ids_train):
            cv_clf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                class_weight="balanced",
                n_jobs=-1,
                min_samples_leaf=2,
                max_features="sqrt",
            )
            cv_clf.fit(X_train[tr], y_train[tr])
            cv_scores.append(float(cv_clf.score(X_train[va], y_train[va])))
        cv_scores_arr = np.array(cv_scores, dtype=np.float32)
        print(f"\nGroup CV ({cv_folds} folds) accuracy: {cv_scores_arr.mean():.4f} ± {cv_scores_arr.std():.4f}")

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nHoldout (group split) accuracy: {acc:.4f}")
    print("\nClassification report:")
    labels = [0, 1, 2]
    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=["Low", "Moderate", "High"],
            zero_division=0,
        )
    )
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=labels))

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_out)
    print(f"\nSaved model to: {model_out}")


def predict_image(model_path: Path, image_path: Path) -> None:
    clf: RandomForestClassifier = joblib.load(model_path)

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise SystemExit(f"Could not read image: {image_path}")

    img = preprocess_image(img_bgr, size=(256, 256))
    feats, t_est = extract_features(img)
    X = np.array([feats], dtype=np.float32)

    if hasattr(clf, "n_features_in_"):
        n_expected = int(clf.n_features_in_)
        if X.shape[1] != n_expected:
            X = X[:, :n_expected]

    _ml_pred = int(clf.predict(X)[0])

    # Calculate density based on the 25% thickest smoke
    t_flat = t_est.ravel()
    k = max(1, int(t_flat.size * 0.25))
    lowest_t = np.partition(t_flat, k)[:k]
    smoke_density = 1.0 - float(np.mean(lowest_t))

    # --- TEXTURE/VARIANCE RULE ---
    threshold_t = float(np.max(lowest_t))
    smoke_mask = (t_est <= threshold_t).astype(np.uint8)
    
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    mean_val, std_val = cv2.meanStdDev(gray, mask=smoke_mask)
    
    brightness = float(mean_val[0][0])
    std = float(std_val[0][0])

    if brightness > 140.0 and std < 20.0:
        smoke_density = 0.0

    smoke_pct = 100.0 * smoke_density

    if smoke_pct < 25.0:
        pred = 0
    elif smoke_pct <= 60.0:
        pred = 1
    else:
        pred = 2

    print(f"Predicted air quality: {label_to_name(pred)}")
    print(f"Smoke density: {smoke_pct:.2f}%")

    # Display: original + estimated transmission
    img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Haze/Smoke Image")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Estimated Transmission\nSmoke {smoke_pct:.1f}% | {label_to_name(pred)}")
    plt.imshow(t_est, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke Density Estimation and Air Quality Monitoring using Dark Channel Prior (DCP)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train RandomForest on RESIDE ITS (haze/trans).")
    p_train.add_argument("--dataset_dir", type=Path, default=Path("dataset"), help="Folder containing haze/ trans/ clear/")
    p_train.add_argument("--limit", type=int, default=-1, help="Max paired samples to load (use -1 for all).")
    p_train.add_argument("--test_size", type=float, default=0.2)
    p_train.add_argument("--random_state", type=int, default=42)
    p_train.add_argument("--n_estimators", type=int, default=600)
    p_train.add_argument("--cv_folds", type=int, default=5, help="GroupKFold CV on train split (0 or 1 to disable).")
    p_train.add_argument("--model_out", type=Path, default=Path("outputs/model.joblib"))

    p_pred = sub.add_parser("predict", help="Predict air quality for a new image.")
    p_pred.add_argument("--model", type=Path, default=Path("outputs/model.joblib"))
    p_pred.add_argument("--image", type=Path, required=True)

    args = parser.parse_args()

    if args.cmd == "train":
        limit = None if args.limit == -1 else args.limit
        train_model(
            dataset_dir=args.dataset_dir,
            limit=limit,
            test_size=args.test_size,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            cv_folds=args.cv_folds,
            model_out=args.model_out,
        )
        return

    if args.cmd == "predict":
        predict_image(args.model, args.image)
        return


if __name__ == "__main__":
    main()