"""Streamlit interface for the forensic pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "src" / "pipeline.py"


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_pipeline_cli(input_path: Path, outdir: Path, modules: Dict[str, bool], settings: Dict[str, str]) -> str:
    cmd = [
        sys.executable,
        str(PIPELINE),
        "--input",
        str(input_path),
        "--outdir",
        str(outdir),
    ]
    flag_map = {
        "detection": "--run-detection",
        "tracking": "--run-tracking",
        "anpr": "--run-anpr",
        "face": "--run-face",
        "tamper": "--run-tamper",
        "package": "--package",
    }
    for key, enabled in modules.items():
        if enabled and key in flag_map:
            cmd.append(flag_map[key])
    if settings.get("detection_model"):
        cmd.extend(["--detection-model", settings["detection_model"]])

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logs: List[str] = []
    for line in process.stdout:  # type: ignore[attr-defined]
        logs.append(line.rstrip())
        st.session_state.setdefault("run_logs", []).append(line.rstrip())
    process.wait()
    if process.returncode != 0:
        logs.append(f"Pipeline exited with status {process.returncode}")
    return "\n".join(logs)


def page_home() -> None:
    st.title("Comprehensive Computer Vision Toolkit")
    st.markdown(
        """
        This interface orchestrates preprocessing, detection, tracking and
        intelligence extraction modules for multimedia evidence.  Refer to
        the README for legal and privacy considerations.
        """
    )


def page_settings() -> None:
    st.header("Settings")
    st.selectbox(
        "Detection model",
        options=["yolov8n.pt", "yolov8s.pt", "custom"],
        key="detection_model",
        help="Select a YOLO checkpoint or provide the path to a custom model",
    )
    st.text_input("Custom model path", key="custom_model_path")


def page_upload() -> None:
    st.header("Upload input")
    uploaded = st.file_uploader("Upload video file", type=["mp4", "mov", "avi"])
    if uploaded:
        uploads_dir = ROOT / "data" / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        save_path = uploads_dir / uploaded.name
        save_path.write_bytes(uploaded.read())
        st.session_state["input_path"] = str(save_path)
        st.success(f"Uploaded to {save_path}")


def page_run() -> None:
    st.header("Run analysis")
    input_path = Path(st.session_state.get("input_path", ""))
    if not input_path.exists():
        st.warning("Upload or specify an input file first.")
        return
    outdir = ROOT / "outputs" / "streamlit_session"
    outdir.mkdir(parents=True, exist_ok=True)

    st.session_state.setdefault("modules", {
        "detection": True,
        "tracking": True,
        "anpr": False,
        "face": False,
        "tamper": True,
        "package": True,
    })

    modules = st.session_state["modules"]
    cols = st.columns(3)
    modules["detection"] = cols[0].checkbox("Detection", value=modules["detection"])
    modules["tracking"] = cols[1].checkbox("Tracking", value=modules["tracking"])
    modules["anpr"] = cols[2].checkbox("ANPR", value=modules["anpr"])
    cols2 = st.columns(3)
    modules["face"] = cols2[0].checkbox("Face", value=modules["face"])
    modules["tamper"] = cols2[1].checkbox("Tamper", value=modules["tamper"])
    modules["package"] = cols2[2].checkbox("Packaging", value=modules["package"])

    if st.button("Run pipeline"):
        st.session_state["run_logs"] = []
        with st.spinner("Running pipeline..."):
            logs = run_pipeline_cli(input_path, outdir, modules, st.session_state)
        st.text_area("Pipeline logs", logs, height=200)
        st.session_state["last_output"] = str(outdir)

    if "run_logs" in st.session_state:
        st.text_area("Latest logs", "\n".join(st.session_state["run_logs"]), height=200)


def page_results() -> None:
    st.header("Results summary")
    outdir = Path(st.session_state.get("last_output", ROOT / "outputs"))
    if not outdir.exists():
        st.info("No results yet. Run the pipeline first.")
        return
    detections = load_json(outdir / "detection" / "detections.json")
    tracks = load_json(outdir / "tracking" / "tracks.json")
    anpr = load_json(outdir / "anpr" / "results.json")
    faces = load_json(outdir / "face_recog" / "results.json")
    tamper = load_json(outdir / "tamper" / "tamper_report.json")

    st.subheader("Detections")
    st.json(detections)
    st.subheader("Tracks")
    st.json(tracks)
    st.subheader("ANPR")
    st.json(anpr)
    st.subheader("Face recognition")
    st.json(faces)
    st.subheader("Tamper report")
    st.json(tamper)


def page_evidence() -> None:
    st.header("Evidence packaging")
    outdir = Path(st.session_state.get("last_output", ROOT / "outputs"))
    manifest_path = outdir / "package_manifest.json"
    if manifest_path.exists():
        st.download_button("Download manifest", manifest_path.read_bytes(), file_name="package_manifest.json")
    report_path = outdir / "report.pdf"
    if report_path.exists():
        st.download_button("Download PDF report", report_path.read_bytes(), file_name="report.pdf")
    st.markdown("Use the packaging module to assemble archives ready for disclosure.")


def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("", ["Home", "Settings", "Upload", "Run Analysis", "Results", "Evidence"])
    if page == "Home":
        page_home()
    elif page == "Settings":
        page_settings()
    elif page == "Upload":
        page_upload()
    elif page == "Run Analysis":
        page_run()
    elif page == "Results":
        page_results()
    elif page == "Evidence":
        page_evidence()


if __name__ == "__main__":  # pragma: no cover
    main()
