# Lego Brick Inspection — Computer Vision Pipeline

A classical computer vision pipeline for industrial-style inspection of Lego bricks: camera intrinsic calibration with reprojection-error model selection, distortion correction, HSV color segmentation, Hough-circle stud detection, contour-based size classification with pixel-to-mm conversion via chessboard extrinsics, and kit matching against a reference catalog with automatic fault diagnosis.

Built with **OpenCV** and **Python**. No deep learning — the entire pipeline is classical CV, end-to-end.

---

## Context

Coursework project for the Computer Vision course (VCOMP) at FEUP, University of Porto, 2026.

The premise: a factory inspection camera looks down at a tray of Lego bricks. The system must (1) detect each brick, (2) identify its color and size, (3) verify whether the tray matches one of three pre-defined kits, and (4) flag faulty kits along with the cause (missing brick, surplus brick, wrong type).

```
┌────────────────────────────────────────────────────────────────┐
│  Raw image                                                     │
│      │                                                         │
│      ▼                                                         │
│  Intrinsic calibration ── 3 calib sets, pick lowest reproj.    │
│      │                    error (best: k₁ ≈ −0.43, strong      │
│      │                    barrel distortion)                   │
│      ▼                                                         │
│  Undistort + ROI crop ── cv.getOptimalNewCameraMatrix          │
│      │                                                         │
│      ▼                                                         │
│  Extrinsic calibration ── solvePnP on chessboard               │
│      │                    → pixel-to-mm scale factor           │
│      ▼                                                         │
│  HSV color segmentation ── multi-range masks per color         │
│      │                    (handles lighting variation)         │
│      ▼                                                         │
│  Contour extraction ── filter by min area, find centroids      │
│      │                                                         │
│      ▼                                                         │
│  Size classification ── area in mm² → {2x1, 2x2, R2x2,         │
│      │                  2x4, 2x6} via reference dimensions     │
│      ▼                                                         │
│  Stud detection ── HoughCircles (cross-check size)             │
│      │                                                         │
│      ▼                                                         │
│  Kit matching ── (color, size, count) signature →              │
│                  closest pre-defined kit by L1 distance →      │
│                  exact match / faulty + cause                  │
└────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Details

### 1. Intrinsic calibration with model selection
Three independent calibration sets (34 chessboard images each, 12×9 inner corners, 15 mm squares) are processed with `cv.findChessboardCorners` + `cv.cornerSubPix` + `cv.calibrateCamera`. The set with the lowest reprojection error is selected — this turns out to have a strong barrel distortion (k₁ ≈ −0.43), which materially affects downstream measurements and is the reason undistortion isn't optional here.

### 2. Extrinsic calibration and pixel-to-mm
`final_setup.png` (the inspection-arrangement image) is undistorted with the best intrinsic matrix, then `cv.solvePnP` recovers the extrinsic matrix from the chessboard. The mean horizontal corner spacing in the undistorted image gives the pixel-to-mm conversion factor used for all later area measurements.

A subtle point handled correctly here: after `getOptimalNewCameraMatrix`, the principal point is shifted, so corner coordinates and any later size computation must stay in the new (undistorted) coordinate frame — no cropping with the returned ROI before `solvePnP`.

### 3. Color segmentation (HSV, multi-range)
Single-range HSV masks fail under uneven lighting — bricks of the "same" color shift noticeably across an image. Each color uses two HSV ranges OR'd together (one for the brighter side, one for the shadowed side):

```python
blue1   = ([100, 124,  85], [110, 255, 255])
blue2   = ([105, 180,  50], [115, 255, 255])  # darker variant
green1  = ([ 42,  95,  60], [ 49, 255, 255])
green2  = ([ 34,  80, 150], [ 44, 255, 255])  # lighter variant
red1    = ([  1, 150,  40], [ 10, 255, 150])
red2    = ([171, 150,  40], [180, 255, 150])  # red wraps around H=0
yellow1 = ([ 13, 145,   1], [ 21, 255, 255])
yellow2 = ([ 22, 120, 160], [ 25, 255, 255])
```

The red ranges also handle the H-wrap at 0/180. Ranges were tuned empirically from the supplied dataset.

### 4. Contour-based size classification
After masking, contours are extracted and filtered by a minimum area threshold (500 px²) to drop noise. Each contour's area is converted to mm² using the calibration factor, then matched to one of the five reference brick footprints (2x1, 2x2, R2x2, 2x4, 2x6) by closest area.

### 5. Stud detection (cross-check)
`HoughCircles` is run on grayscale crops to count studs per brick, providing an independent check on the area-based size classification.

### 6. Kit matching and fault diagnosis
Each detected scene is reduced to a `(color, size) → count` signature. For inspection images A, B, C, the L1 distance is computed against each pre-defined kit signature (Kit 1, Kit 2, Kit 3); the closest is reported as a match (`diff = 0`) or as a fault with the diff vector indicating which bricks are missing or in surplus.

---

## What's Hard About This Problem

The instinct is "it's just colored blocks on a white background, segment by color and you're done." The actual difficulties:

- **Strong barrel distortion** (k₁ ≈ −0.43). Bricks far from the principal point measure significantly smaller in pixels than identical bricks at the center if you skip undistortion. Without correction, size classification is unreliable across the image.
- **Lighting non-uniformity.** Each color needs at least two HSV ranges to handle bright vs. shadowed regions of the same brick.
- **Reference frame consistency.** The pixel-to-mm factor is only valid in the undistorted coordinate system with the new camera matrix — mixing it with raw image coordinates silently breaks the size classification.
- **R2x2 vs. 2x2 disambiguation.** Same area, different shape (R2x2 is rectangular). Raw area thresholding isn't enough; aspect ratio of the contour bounding box is used as a tie-breaker.

---

## Repository Layout

```
.
├── notebooks/
│   └── main.ipynb           # End-to-end pipeline, all 4 tasks
├── results/                 # Saved output figures (detection overlays,
│                            # calibration tables, kit-match results)
├── requirements.txt
├── .gitignore
└── README.md
```

The original `data/` folder (calibration chessboards, isolated brick images, kit images, fault images) is **not** included — those images are course material and aren't redistributed here. The notebook expects:

```
data/
├── Calibration/
│   ├── calib1/calib_0.png … calib_33.png
│   ├── calib2/…
│   ├── calib3/…
│   └── final_setup.png
├── Isolated/
│   ├── colored_bricks.png
│   ├── blue.png · green.png · red.png · yellow.png
├── Kit/
│   ├── kit1.png · kit2.png · kit3.png
│   └── ImageA_kit.png · ImageB_kit.png · ImageC_kit.png
└── Fault/
    └── fault_*.png
```

---

## Running

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/main.ipynb
```

Run cells top to bottom. The pipeline is self-contained and produces the result tables and detection overlay figures inline.

---

## Tooling

- Python 3.11
- OpenCV 4.x (`cv2`)
- NumPy, Pandas
- Matplotlib (visualization)
- Jupyter

---

## Author

Lars Husemann — M.Sc. Electrical and Computer Engineering, TU Munich (Erasmus semester at FEUP, University of Porto).
