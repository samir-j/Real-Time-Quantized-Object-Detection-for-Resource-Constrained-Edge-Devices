"""
download_dataset.py
-------------------
Downloads the PPE / Hard Hat Workers dataset from Roboflow Universe.

Prerequisites:
    pip install roboflow

Usage:
    python scripts/download_dataset.py --api-key YOUR_ROBOFLOW_API_KEY

If you don't have a Roboflow account, you can also download directly from:
  • Roboflow Universe (free account):
    https://universe.roboflow.com/joseph-nelson/hard-hat-workers
  • Kaggle (requires kaggle CLI):
    https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection
  • Direct ZIP (no login):
    See README.md for the Google Drive mirror link.
"""

import argparse
import os
import sys
import zipfile
import shutil
from pathlib import Path

def download_via_roboflow(api_key: str, save_dir: str = "./data"):
    """Download dataset using the official Roboflow Python SDK."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] roboflow package not found. Run: pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("joseph-nelson").project("hard-hat-workers")
    version = project.version(12)                    # latest stable version
    dataset = version.download("yolov8", location=f"{save_dir}/ppe_dataset")

    print(f"\n[OK] Dataset downloaded to: {dataset.location}")
    print(f"     Splits   : train / valid / test")
    return dataset.location


def download_via_kaggle(save_dir: str = "./data"):
    """
    Download via Kaggle CLI.
    Requires: pip install kaggle  +  ~/.kaggle/kaggle.json credentials
    """
    import subprocess
    dest = Path(save_dir) / "ppe_dataset"
    dest.mkdir(parents=True, exist_ok=True)

    cmd = [
        "kaggle", "datasets", "download",
        "-d", "andrewmvd/hard-hat-detection",
        "-p", str(dest), "--unzip"
    ]
    print("[INFO] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[OK] Dataset extracted to: {dest}")
    return str(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PPE detection dataset")
    parser.add_argument("--api-key", type=str, default="",
                        help="Roboflow API key (get free at app.roboflow.com)")
    parser.add_argument("--source", choices=["roboflow", "kaggle"], default="roboflow",
                        help="Dataset source (default: roboflow)")
    parser.add_argument("--save-dir", type=str, default="./data",
                        help="Directory to save dataset (default: ./data)")
    args = parser.parse_args()

    print("=" * 60)
    print("  PPE Detection — Dataset Downloader")
    print("=" * 60)

    if args.source == "roboflow":
        if not args.api_key:
            print("[ERROR] --api-key is required for Roboflow download.")
            print("        Get a free key at: https://app.roboflow.com")
            sys.exit(1)
        loc = download_via_roboflow(args.api_key, args.save_dir)
    else:
        loc = download_via_kaggle(args.save_dir)

    print("\n[DONE] Update data/ppe.yaml → path: to point at:", loc)
