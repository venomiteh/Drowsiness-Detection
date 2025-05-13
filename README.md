# Drowsiness Detection System

This project is a real-time drowsiness detection system that uses a combination of **computer vision**, **deep learning (CNN)**, and **MediaPipe facial landmark detection** to monitor a user's eye and mouth states for signs of sleepiness. If signs of drowsiness are detected—such as prolonged eye closure, yawning, or head nodding—an alarm is triggered to alert the user.

## 🔍 Features

- Real-time webcam-based detection
- Eye state classification (open/closed)
- Yawn detection using mouth state classification
- Head-nod (tilt) detection
- Alerts using audio alarms (different sounds for eyes and yawns)
- Utilizes:
  - PyTorch-trained CNN models (`eye_cnn_model.pth` and `yawn_cnn_model.pth`)
  - MediaPipe for facial landmark tracking
  - Pygame for sound alerts

## 🧠 Models Used

- Two custom CNN models (`ProjCNN`) trained separately:
  - One for detecting open/closed eye states
  - One for detecting yawning (mouth states)
- Input to each model is grayscale, normalized, and resized to 86x86 pixels

## 🛠 Technologies

- Python
- OpenCV
- MediaPipe
- PyTorch
- Pygame
- TorchVision

## 🗂 Files

| File Name            | Description                                        |
|---------------------|----------------------------------------------------|
| `DrowsinessDetection.py` | Main detection script; loads models, captures video, detects drowsiness |
| `eye_cnn_model.pth`     | Pretrained model for eye state detection         |
| `yawn_cnn_model.pth`    | Pretrained model for yawn (mouth) detection      |
| `eyemodel.ipynb`        | Jupyter notebook for training/testing eye model  |
| `yawn-model.ipynb`      | Jupyter notebook for training/testing yawn model |
| `warning-alarm.mp3`     | Audio alert for drowsy eyes                      |
| `alarm_voice.wav`       | Audio alert for yawning                         |

## ▶️ How to Run

1. Make sure you have a webcam connected.
2. Install the required libraries:
   ```bash
   pip install opencv-python torch torchvision mediapipe pygame
Place the .pth files and alarm sound files in the same directory as the script.

Run the script:

bash
Copy
Edit
python DrowsinessDetection.py
Press q to exit the application.

## ⚠️ Notes
Ensure good lighting for more accurate landmark detection.

Model accuracy may vary depending on training quality and input resolution.

The system is designed for a single face in the camera frame.
