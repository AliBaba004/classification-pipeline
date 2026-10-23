# Classification Pipeline

On-device lighting classification project for **Project I - Assignment 1**.

The system uses an ESP32 with UV and IR sensor measurements to classify four lighting conditions:

- LED
- Fluorescent
- Shade
- Sun

## Pipeline

The model is trained offline using:

1. EMA filtering
2. Log transformation and feature scaling
3. DBSCAN clustering
4. Gaussian KDE
5. 16 × 16 density-map sampling
6. `uint8` quantization
7. Lookup-table inference on the ESP32

DBSCAN and KDE are used during training. The ESP32 performs preprocessing and table lookup during live classification.

## Hardware

- ESP32 development board
- Analog UV sensor on GPIO34
- TSL2591 light sensor
- I2C:
  - SDA: GPIO21
  - SCL: GPIO22

The TSL2591 IR channel is used as the second classification feature.

## Dataset

The labelled dataset contains 2400 measurements:

| Class | Samples |
|---|---:|
| LED | 600 |
| Fluorescent | 600 |
| Shade | 600 |
| Sun | 600 |
| Total | 2400 |

The dataset was split into:

- 60% training
- 20% validation
- 20% test

## Final Model

Final model settings:

- DBSCAN `eps = 0.02`
- DBSCAN `min_samples = 8`
- KDE bandwidth `h = 0.03`
- Grid size: `16 × 16`
- Number of classes: 4
- Reject threshold: `0.12`

The four lookup tables require 1024 bytes.

The scaling bounds add 16 bytes.

Total stored model data:

`1040 bytes`

Measured worst-case `uint8` quantization error:

`0.193%`

## Results

The original held-out test contained 480 samples and produced 480 correct classifications.

A separately collected deployment batch contained 2324 readings. The ESP32 and desktop lookup implementations agreed on all 2324 readings.

For the fresh deployment batch, LED, fluorescent, and shade were classified correctly. The outdoor batch was collected under overcast conditions and was rejected as UNKNOWN rather than being incorrectly classified as clear sun.

The final ESP32 program used:

- Flash: `308800 bytes`
- SRAM: `23740 bytes`
- Classification latency: approximately `3-6 µs` per query

## Repository Structure

```text
classification-pipeline/
├── assignment1/
│   ├── data/
│   ├── firmware/
│   │   ├── on_device_inference.ino
│   │   └── model.h
├── figures/
│   └── generated/
├── paper/
│   ├── paper.tex
│   └── classification_pipeline_paper.pdf
├── src/
│   ├── retrain_dbscan.py
│   ├── retrain_kde.py
│   ├── generate_real_surfaces.py
│   └── generate_model_header.py
├── .gitignore
└── README.md
```

## Main Files

- `assignment1/data/` - labelled real sensor measurements
- `assignment1/firmware/on_device_inference.ino` - ESP32 classification firmware
- `assignment1/firmware/model.h` - generated embedded lookup model
- `src/retrain_dbscan.py` - DBSCAN training
- `src/retrain_kde.py` - KDE training and bandwidth validation
- `src/generate_real_surfaces.py` - density-map generation
- `src/generate_model_header.py` - generates the embedded C header
- `paper/classification_pipeline_paper.pdf` - final report

## Repository

Project repository:

https://github.com/AliBaba004/classification-pipeline

Final submitted version:

`v2.0`
