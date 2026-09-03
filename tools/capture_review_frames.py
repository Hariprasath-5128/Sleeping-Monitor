"""Save a small sample of ESP32-CAM frames for visual review.

Usage:
    python tools/capture_review_frames.py --cam-ip 10.77.153.2 --port 81
"""

import argparse
from datetime import datetime
from pathlib import Path
import time

import cv2


def main():
    parser = argparse.ArgumentParser(description="Capture ESP32-CAM review frames")
    parser.add_argument("--cam-ip", required=True, help="ESP32-CAM IPv4 address")
    parser.add_argument("--port", type=int, default=81, help="MJPEG stream port")
    parser.add_argument("--count", type=int, default=5, help="Number of frames to save")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between saved frames")
    args = parser.parse_args()

    stream_url = f"http://{args.cam_ip}:{args.port}/stream"
    output_dir = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "review_captures"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    capture = cv2.VideoCapture(
        stream_url,
        cv2.CAP_FFMPEG,
        [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
        ],
    )
    if not capture.isOpened():
        raise SystemExit(f"Could not open stream: {stream_url}")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        deadline = time.monotonic() + 15
        next_capture = time.monotonic()

        while saved < args.count and time.monotonic() < deadline:
            ok, frame = capture.read()
            if not ok:
                time.sleep(0.05)
                continue

            if time.monotonic() < next_capture:
                continue

            path = output_dir / f"frame_{saved + 1:02d}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise SystemExit(f"Could not write: {path}")
            print(path)
            saved += 1
            next_capture = time.monotonic() + args.interval

        if saved != args.count:
            raise SystemExit(f"Saved {saved}/{args.count} frames before the stream timed out.")
    finally:
        capture.release()

    print(f"Saved {saved} frames to: {output_dir}")


if __name__ == "__main__":
    main()
