# Comprehensive Computer Vision Toolkit (CCT)

The Comprehensive Computer Vision Toolkit is a modular pipeline for
multimedia forensics.  It ingests a video file, extracts evidential
frames, performs object detection and tracking, recognises number plates
and faces, and packages the results into a courtroom-friendly report.
The repository is intended as a strong baseline that investigators can
extend with domain specific models.

## Prerequisites

* Python 3.10+
* Recommended operating systems: Linux or Windows 10/11.  macOS works
  for CPU-only workflows.
* System packages: `ffmpeg`, `tesseract-ocr`.  When running face
  recognition you may need `cmake` and a C++ build chain to install
  `face_recognition`/`dlib`.
* Optional GPU support requires recent NVIDIA drivers and CUDA 12.1.

Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

For Windows users struggling with `face_recognition`, consider
substituting [InsightFace](https://github.com/deepinsight/insightface) or
[DeepFace](https://github.com/serengil/deepface) and adapting the face
pipeline.  The code already catches missing dependencies and reports
helpful errors.

## Repository layout

```
src/
  forensics/preprocess.py         # frame extraction + hashing
  detection/yolo_infer.py         # YOLO inference wrapper
  tracking/simple_tracker.py      # baseline IoU tracker
  tracking/appearance_tracker.py  # ResNet18 feature tracker
  tracking/export_crops_and_visualize.py
  anpr/anpr_pipeline.py           # number plate extraction + OCR
  anpr/aggregate_anpr_results.py  # per-track aggregation
  vehicle_recognition/            # ImageNet baseline for vehicles
  face_recog/gallery_manager.py   # build gallery from known faces
  face_recog/face_recog_pipeline.py
  tamper/tamper_detector.py       # hashing, metadata, PRNU, deepfake stub
  packager/manifest.py            # manifest builder
  packager/pdf_report.py          # courtroom style PDF report
  pipeline.py                     # orchestrator
streamlit_app/app.py              # Streamlit UI entry point
```

Two example notebooks illustrating preprocessing and detection/tracking
live in the `notebooks/` directory.

## Basic pipeline run

1. Place the evidence video at `data/sample_videos/new_sample.mp4` (or
   provide another path).
2. Run the end-to-end pipeline on CPU:

   ```bash
   python -m src.pipeline \
       --input data/sample_videos/new_sample.mp4 \
       --outdir outputs/case_001 \
       --run-detection --run-tracking --run-anpr --run-face --run-tamper --package
   ```

3. Inspect the generated artefacts inside `outputs/case_001/`:
   * `preprocess/` – extracted frames + hashes
   * `detection/` – YOLO detections and annotated imagery
   * `tracking/` & `crops/` – track JSON, crops and annotated video
   * `anpr/`, `face_recog/`, `tamper/` – specialist analytics
   * `package_manifest.json` & `report.pdf` – final evidence bundle

The `src/pipeline.py` script accepts additional options (frame skipping,
custom YOLO weights, alternative trackers, gallery locations, etc.). Run
`python -m src.pipeline --help` for the full list.

## Adding known faces

1. Create a folder per person inside `data/known_faces/<person>/images`.
2. Populate with clear face photographs (`.jpg`).
3. Rebuild the gallery encodings:

   ```bash
   python -m src.face_recog.gallery_manager
   ```

   This creates `data/gallery_encodings.pkl` which is consumed by the
   face recognition pipeline.

4. Run the pipeline with `--run-face` to apply the gallery to tracked
   person crops.

## Running tests

The repository ships with Pytest-based smoke tests:

```bash
pytest
```

The preprocessing test automatically skips when OpenCV is not available.

## Docker usage

Two Dockerfiles are supplied in `docker/`:

* `Dockerfile.cpu` – CPU-only image based on `python:3.10-slim`.  It
  installs FFmpeg, Tesseract and the Python dependencies.
* `Dockerfile.gpu` – CUDA 12.1 runtime image for NVIDIA GPUs.  Install
  the appropriate CUDA-enabled PyTorch wheel after building the image.

Build the CPU image and run the pipeline:

```bash
docker build -t cct-cpu -f docker/Dockerfile.cpu .
docker run --rm -v $(pwd):/app cct-cpu \
    python -m src.pipeline --input data/sample_videos/new_sample.mp4 --outdir outputs/container_case \
    --run-detection --run-tracking --run-tamper --package
```

For GPU usage make sure `nvidia-docker` is installed and expose the GPU
with `--gpus all`.

## Streamlit UI

Launch the user interface with:

```bash
streamlit run streamlit_app/app.py
```

The UI provides pages for settings, uploads, running the pipeline and
reviewing results.  The run page spawns the orchestrator in a
sub-process and surfaces log output.

## Security and privacy notes

* Always process sensitive evidence locally.  Disable automatic uploads
  in tooling (including Streamlit file upload history).
* The manifest and PDF report capture UTC timestamps, SHA256 hashes,
  module parameters, tool versions, and a chain-of-custody template.
* PRNU and deepfake checks are intentionally conservative placeholders.
  Replace them with production-ready models when required.

## Contact / credits

This toolkit was assembled as a reference implementation for computer
vision forensics teams.  Contributions and bug reports are welcome via
issues and pull requests.  For legal consultation engage qualified
counsel familiar with digital evidence handling.

## Example workflows

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.face_recog.gallery_manager
python -m src.pipeline `
    --input data/sample_videos/new_sample.mp4 `
    --outdir outputs/case_windows `
    --run-detection --run-tracking --run-anpr --run-face --run-tamper --package
```

### Linux / WSL

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.face_recog.gallery_manager
python -m src.pipeline \
    --input data/sample_videos/new_sample.mp4 \
    --outdir outputs/case_linux \
    --run-detection --run-tracking --run-anpr --run-face --run-tamper --package
```

## Quick runbook

1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Copy evidence to `data/sample_videos/new_sample.mp4`
4. (Optional) add known faces and rebuild the gallery
5. `python -m src.pipeline --input data/sample_videos/new_sample.mp4 --outdir outputs/case_quick --run-detection --run-tracking --run-anpr --run-face --run-tamper --package`
6. Review `outputs/case_quick/report.pdf` and share the generated
   `package_manifest.json` with investigators.

## created_files.json

After the repository generation, `created_files.json` summarises the key
assets added by the scaffolding scripts.
