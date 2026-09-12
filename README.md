# CodeAlpha_TrafficMonitor

## Project Overview
This project implements a lightweight, moving-camera traffic monitoring system tailored for edge devices like the Raspberry Pi. Utilizing a CPU-optimized YOLOv8 ONNX detector, it features motion-compensated tracking, constant-velocity prediction, and IoU/center association. The system provides real-time vehicle counting, direction tracking, and optional speed estimation while maintaining low latency and memory footprints.




https://github.com/user-attachments/assets/fc939fd9-eadd-4498-847d-96769bf8c0b4




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

<img width="33%" height="1636" alt="20260911_192547(0)" src="https://github.com/user-attachments/assets/0bac3f4d-6298-4000-a152-ab4309ccc74d" />
<img width="33%" height="1516" alt="20260911_192239" src="https://github.com/user-attachments/assets/dc81b8a2-2e71-4b5e-a30b-3d0868bb2bfd" />
<img width="33%" height="1680" alt="20260911_192547" src="https://github.com/user-attachments/assets/432101f0-d61a-41b8-add7-755f5717b46e" />

# Author
Ahmed Emad Mostafa
