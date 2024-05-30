from pathlib import Path

HERE = Path(__file__).parent

MODEL_PATH = HERE / "./models/"


# This is only needed when hosting in local
# Download from: https://www.gyan.dev/ffmpeg/builds/
FFMPEG_PATH = HERE / "./ffmpeg/ffmpeg.exe"

DEBUG = False

STYLES = {
    "yolov8n": "yolov8n.pt",
    "yolov10n":"yolov10n.pt",
    "yolov9-c": "yolov9-c.pt",
}

