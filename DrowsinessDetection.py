from re import A
import cv2 as cv
import torch as tr
import torch.nn as nn
import torchvision.transforms as transforms
import mediapipe as mp
import threading as th
import pygame




# This is to load the mediapipe face mesh model (The one that detects the face landmarks)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)


# Now we define the landmark indices for the mouth and eyes

MOUTH_LANDMARKS =[ 61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 78, 191, 80,
    81, 82, 13, 312, 311, 310, 415, 308,
    78, 95, 88, 178, 87, 14, 317, 402,
    318, 324, 308, 415, 310, 311, 312, 13,
    82, 81, 80, 191]


LEFT_EYE_LANDMARKS = [
    159, 160, 161, 246, 145, 144, 153, 154, 155,
    133, 33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246
]

RIGHT_EYE_LANDMARKS = [
    386, 385, 384, 398, 374, 373, 380, 381, 382,
    362, 263, 249, 390, 373, 374, 380, 381, 382, 362,
    398, 384, 385, 386, 387, 388, 466
]

print(len(MOUTH_LANDMARKS))



class ProjCNN(nn.Module):
    def __init__(self):
        super(ProjCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 10 * 10, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# Transform for grayscale input
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((86, 86)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


#load Models

model_eye = ProjCNN()
model_eye.load_state_dict(tr.load("eye_cnn_model.pth", map_location=tr.device('cpu')))
model_eye.eval()

model_mouth = ProjCNN()
model_mouth.load_state_dict(tr.load("yawn_cnn_model.pth", map_location=tr.device('cpu')))
model_mouth.eval()


#variables to store the state of the eyes
eye_state = 0

pygame.mixer.init()
alarm_channel_eye = pygame.mixer.Channel(0)
alarm_channel_yawn = pygame.mixer.Channel(1)
eye_or_head_alarm_active = False
# Load sounds once
alarm_sound_eye = pygame.mixer.Sound('warning-alarm.mp3')
alarm_sound_yawn = pygame.mixer.Sound('alarm_voice.wav')
vid = cv.VideoCapture(0)
vid.set(cv.CAP_PROP_FPS, 60)
vid.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
vid.set(cv.CAP_PROP_BRIGHTNESS, 0)  # Value range may vary
vid.set(cv.CAP_PROP_EXPOSURE,-3)    
initial_nose_y = None
drowsy_head_counter = 0
head_drop_threshold = 0.05  # Adjust sensitivity (5% downward movement)

while True:
    ret, frame = vid.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Required for MediaPipe
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[1]
            x_nose, y_nose = int(nose_tip.x * w), int(nose_tip.y * h)
            cv.circle(frame, (x_nose, y_nose), 2, (0, 0, 255), -1)
            # Draw landmarks
            for idx in MOUTH_LANDMARKS + LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS:
                point = face_landmarks.landmark[idx]
                x, y = int(point.x * w), int(point.y * h)
                cv.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Left eye
            left_eye_points = [face_landmarks.landmark[i] for i in LEFT_EYE_LANDMARKS]
            left_coords = [(int(p.x * w), int(p.y * h)) for p in left_eye_points]
            x_min, x_max = min(x for x, y in left_coords), max(x for x, y in left_coords)
            y_min, y_max = min(y for x, y in left_coords), max(y for x, y in left_coords)

            # Right eye
            right_eye_points = [face_landmarks.landmark[i] for i in RIGHT_EYE_LANDMARKS]
            right_coords = [(int(p.x * w), int(p.y * h)) for p in right_eye_points]
            x_min_r, x_max_r = min(x for x, y in right_coords), max(x for x, y in right_coords)
            y_min_r, y_max_r = min(y for x, y in right_coords), max(y for x, y in right_coords)

            #mouth
            mouth_points = [face_landmarks.landmark[i] for i in MOUTH_LANDMARKS]
            mouth_coords = [(int(p.x * w), int(p.y * h)) for p in mouth_points]
            x_min_m, x_max_m = min(x for x, y in mouth_coords), max(x for x, y in mouth_coords)
            y_min_m, y_max_m = min(y for x, y in mouth_coords), max(y for x, y in mouth_coords)


            # Draw bounding boxes
            cv.rectangle(frame, (x_min-20, y_min-20), (x_max+20, y_max+20), (255, 0, 0), 2)
            cv.rectangle(frame, (x_min_r-20, y_min_r-20), (x_max_r+20, y_max_r+20), (255, 0, 0), 2)

            # Crop eye regions
            left_eye_img = frame[y_min-20:y_max+20, x_min-20:x_max+20]
            right_eye_img = frame[y_min_r-20:y_max_r+20, x_min_r-20:x_max_r+20]
            mouth_img = frame[y_min_m:y_max_m+20, x_min_m-20:x_max_m+20]

            # Convert to grayscale
            left_eye_gray = cv.cvtColor(left_eye_img, cv.COLOR_BGR2GRAY)
            right_eye_gray = cv.cvtColor(right_eye_img, cv.COLOR_BGR2GRAY)
            mouth_img = cv.cvtColor(mouth_img, cv.COLOR_BGR2GRAY)

            # --- Head Tilt Detection ---
            current_nose_y = nose_tip.y
            

            if initial_nose_y is None:
                initial_nose_y = current_nose_y  # Set the initial nose y position

# Check how much the nose dropped
            if current_nose_y - initial_nose_y > head_drop_threshold:
                drowsy_head_counter += 1
            else:
                drowsy_head_counter = 0
            
            

# If head stayed down for enough frames
            if drowsy_head_counter > 5:
                if not alarm_channel_eye.get_busy():
                    alarm_channel_eye.play(alarm_sound_eye, loops=-1)
                    eye_or_head_alarm_active = True
                  

# Optional: show info on screen
            cv.putText(frame, f'Head Tilt: {"Down" if drowsy_head_counter > 0 else "Normal"}', (30, 130), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


            try:
                input_tensor1 = transform(left_eye_gray).unsqueeze(0)
                input_tensor2 = transform(right_eye_gray).unsqueeze(0)
                input_tensor3 = transform(mouth_img).unsqueeze(0)


                with tr.no_grad():
                    out1 = model_eye(input_tensor1)
                    out2 = model_eye(input_tensor2)
                    out3 = model_mouth(input_tensor3)

                pred1 = tr.argmax(out1, dim=1).item()
                pred2 = tr.argmax(out2, dim=1).item()
                pred3 = tr.argmax(out3, dim=1).item()
                if pred1 == 0 and pred2 == 0:  # both eyes closed
                    eye_state += 1
                else:
                    eye_state = 0

                if eye_state > 5:
                    if not alarm_channel_eye.get_busy():
                        alarm_channel_eye.play(alarm_sound_eye, loops=-1)
                        eye_or_head_alarm_active = True
                
                if drowsy_head_counter <= 5 and eye_state <= 5:
                    if alarm_channel_eye.get_busy():
                        alarm_channel_eye.stop()
                        eye_or_head_alarm_active = False

                

                        
                    
                    
                if pred3 == 1:
                    if not alarm_channel_yawn.get_busy():
                        alarm_channel_yawn.play(alarm_sound_yawn)
                        
                    
                    

                label_map = {0: 'Closed_Eyes', 1: 'Open_Eyes'}
                label_map_mouth = {0: 'no yawn', 1: 'yawn'}
                cv.putText(frame, f'Eyes: {"Closed" if pred1 == 0 and pred2 == 0 else "Open"}', (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv.putText(frame, f'Mouth: {"Yawning" if pred3 == 1 else "Normal"}', (30, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            except Exception as e:
                print("Error processing eye region:", e)

    cv.imshow('Eye Detection', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()





