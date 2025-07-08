
# Cross-Camera Player Mapping

## Setup Instructions

### 1. Environment Setup

Make sure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Download Model

Download the YOLOv11 model from [this link](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view) and place it in the root directory as `yolov11_custom.pt`.

### 3. Run the Code

```bash
python src/player_mapping.py
```

## Dependencies

- ultralytics
- opencv-python
- numpy
- deep-sort-realtime

To install dependencies manually:

```bash
pip install ultralytics opencv-python numpy deep-sort-realtime
```
