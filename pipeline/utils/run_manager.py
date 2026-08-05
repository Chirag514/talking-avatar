"""
run_manager.py

Creates and tracks one "run folder" per video job, with a manifest.json
that records which stage ran, with what parameters, how long it took,
and where its output landed.

RUNPOD STORAGE MODEL:
Rented GPU pods (RunPod and similar) typically expose two different
disks:
  - Local container disk (e.g. under /root or /tmp) — fast NVMe,
    but WIPED if the pod is terminated/deleted (not just stopped).
  - Persistent network volume (conventionally mounted at /workspace) —
    slower (network-attached), but survives pod stop/restart/termination
    as long as the volume itself isn't deleted.

This mirrors the Colab "local disk vs Google Drive" split from the
original version of this file, just with different mount points and a
different reason for the split (network-attached latency rather than a
FUSE-mounted consumer cloud drive). All working files during a run use
local container disk; call sync_to_persistent_storage() once at the end
to persist the final result before you stop/terminate the pod.
"""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path


class RunManager:
    """
    One instance per video job. Owns a folder like:

        /root/pipeline_runs/2026-07-09_template03_run1/
            00_frames/
            02_voice.wav
            04_lipsync_raw.mp4
            05_restored.mp4
            05_unrestored.mp4
            06_final.mp4
            manifest.json

    All of the above lives on fast LOCAL container disk during
    processing. Call sync_to_persistent_storage() at the end to copy
    results to the persistent network volume (/workspace) so they
    survive a pod stop/restart/termination.
    """

    def __init__(self, run_name: str, base_dir: str = "/root/pipeline_runs"):
        # Local container disk by default — fast, but ephemeral. Pass
        # base_dir="/workspace/pipeline_runs" explicitly if you want to
        # skip the local-scratch/sync-at-the-end pattern entirely (e.g.
        # a quick one-off debug run where losing intermediate files on
        # pod termination doesn't matter and you'd rather not manage
        # two locations).
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.run_id = f"{timestamp}_{run_name}"
        self.run_dir = Path(base_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.run_dir / "manifest.json"
        self.manifest = {
            "run_id": self.run_id,
            "created_at": timestamp,
            "stages": {},
        }
        self._save_manifest()

    def path_for(self, name: str) -> Path:
        """Returns an absolute path inside this run's folder for a given
        stage output file or subfolder name, e.g. '00_frames' or '02_voice.wav'."""
        p = self.run_dir / name
        return p

    def stage_input_locally(self, source_path: str) -> Path:
        """
        Copies an input file (template video, reference voice/face clip)
        from the persistent network volume (/workspace) into this run's
        fast local working folder, and returns the new local path.

        Call this on every /workspace-sourced input BEFORE passing it
        into a stage function, so stages like Stage 0's ffmpeg frame
        extraction read from local disk instead of the network volume.
        If the input is already on local disk (doesn't start with
        /workspace/), it's returned unchanged — no wasted copy.
        """
        source_path = Path(source_path)
        if "/workspace/" not in str(source_path):
            return source_path  # already local, nothing to do

        local_dir = self.path_for("00_inputs")
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / source_path.name
        shutil.copy2(source_path, local_path)
        return local_path

    def record_stage(self, stage_name: str, params: dict, output_path: str,
                      duration_seconds: float, status: str = "ok", notes: str = ""):
        """Call this immediately after a stage finishes, success or fail."""
        self.manifest["stages"][stage_name] = {
            "params": params,
            "output_path": str(output_path),
            "duration_seconds": round(duration_seconds, 2),
            "status": status,
            "notes": notes,
            "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save_manifest()

    def _save_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def total_duration(self) -> float:
        return sum(s["duration_seconds"] for s in self.manifest["stages"].values())

    def sync_to_persistent_storage(self, persistent_base_dir: str = "/workspace/pipeline_runs",
                                    full_run: bool = False) -> Path:
        """
        Copies results to the persistent network volume. Call this once,
        at the very end of a run, before you stop or terminate the pod —
        anything left only on local container disk is lost when the pod
        is terminated (deleted), though a simple "stop" on some providers
        preserves container disk too. Don't rely on that distinction;
        always sync explicitly.

        full_run=False (default): copies only 06_final.mp4 and
        manifest.json — the two things you actually need to keep. Fast,
        since it skips hundreds of intermediate frame PNGs and the
        unrestored/lipsync-raw intermediate videos.

        full_run=True: copies the entire run folder, including every
        intermediate file — useful for deep debugging a specific run,
        but noticeably slower since it re-introduces the same
        many-small-files network-volume overhead this split was meant
        to avoid. Use sparingly, not as your default.
        """
        persistent_run_dir = Path(persistent_base_dir) / self.run_id
        persistent_run_dir.mkdir(parents=True, exist_ok=True)

        if full_run:
            shutil.copytree(self.run_dir, persistent_run_dir, dirs_exist_ok=True)
        else:
            shutil.copy2(self.manifest_path, persistent_run_dir / "manifest.json")
            final_video = self.path_for("06_final.mp4")
            if final_video.exists():
                shutil.copy2(final_video, persistent_run_dir / "06_final.mp4")

        return persistent_run_dir


class Timer:
    """Tiny context manager so every stage script reports its own wall-clock time
    without repeating time.time() boilerplate."""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
