import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import mediapipe as mp
import mediapipe.tasks
import mediapipe.tasks.python
import mediapipe.tasks.python.vision
import torch
import numpy as np
from train_model import CNN1D_ChordClassifier


def main():
    checkpoint = torch.load('data/models/best_chord_model.pth', map_location='cpu')
    le = checkpoint['label_encoder']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN1D_ChordClassifier(checkpoint['num_chords']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model — {checkpoint['num_chords']} chords: {le.classes_}")
    print("Press 'q' to quit")

    model_path = os.path.join(os.path.dirname(__file__), '..', 'hand_landmarker_f16.task')
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1,
        )
    )

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect(mp_image)

        if results.hand_landmarks:
            h, w = frame.shape[:2]
            for hand in results.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                for i, (x, y) in enumerate(pts):
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            lm_vec = []
            for lm in results.hand_landmarks[0]:
                lm_vec.extend([lm.x, lm.y, lm.z])

            with torch.no_grad():
                out   = model(torch.FloatTensor(lm_vec).unsqueeze(0).to(device))
                probs = torch.softmax(out, dim=1)
                conf, pred = probs.max(1)

            chord = le.inverse_transform([pred.item()])[0]
            pct   = conf.item() * 100
            color = (0, 255, 0) if pct > 80 else (0, 165, 255)

            cv2.putText(frame, f"Chord: {chord}",          (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.putText(frame, f"Confidence: {pct:.1f}%",  (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,   color, 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Guitar Chord Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
