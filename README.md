# CodeAlpha_TrafficMonitor

## Project Overview
This project implements a lightweight, moving-camera traffic monitoring system tailored for edge devices like the Raspberry Pi. Utilizing a CPU-optimized YOLOv8 ONNX detector, it features motion-compensated tracking, constant-velocity prediction, and IoU/center association. The system provides real-time vehicle counting, direction tracking, and optional speed estimation while maintaining low latency and memory footprints.

## Hardware Setup
* **Processing Unit:** Raspberry Pi 4 (Recommended for optimal CPU-only inference) or equivalent edge device.
* **Camera:** Standard USB Webcam or Raspberry Pi Camera Module.
* **Accessories:** MicroSD card (with Raspberry Pi OS installed) and a reliable 5V/3A power supply.
* **Physical Placement:** Connect the camera to the Raspberry Pi, mount it overlooking the roadway, and ensure the focal plane and daylight illumination are clear for tracking.

## Environment & Execution Setup
Run the following commands sequentially to initialize the workspace, install dependencies, export the model to ONNX, and start monitoring:

# 1. Initialize workspace and virtual environment
```bash
mkdir -p ~/traffic
cd ~/traffic
python3 -m venv ~/venv
source ~/venv/bin/activate
```
# 2. Install dependencies
```bash
pip install -r requirements.txt
```
# 3. Export YOLOv8 to ONNX format
```bash
yolo export model=yolov8n.pt format=onnx imgsz=320
```
# 4. Run the traffic monitor
```bash
python traffic_monitor_v2.py \
  --model yolov8n.onnx \
  --source 0 \
  --line 360 \
  --conf 0.25 \
  --save-video live_traffic_v2.mp4 \
  --trails
```
# Project Limitations & Future Work
Processing Power Constraints: The system currently runs on a Raspberry Pi, which limits the overall processing speed and frame rate capabilities.

High-Speed Tracking: Because of the low processing power, the model struggles to keep up with fast-moving objects. While it accurately tracks smaller, slow-moving pedestrians, it frequently drops tracking on motorcycles solely due to their high speed.

Camera Hardware Limits: Tracking accuracy drops at night or in low-light conditions because the current standard camera cannot capture clear footage without daylight.

Proposed Solutions: Future improvements will rely on hardware upgrades, specifically integrating a high-quality camera with built-in night vision to resolve low-light issues, and utilizing a more powerful processing unit to accurately track high-velocity vehicles.


# Author
Ahmed Emad Mostafa
