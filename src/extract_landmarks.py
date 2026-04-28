import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

# Tasks API (mediapipe 0.10+) — IMAGE mode is synchronous, suitable for static images
_model_path = os.path.join(os.path.dirname(__file__), '..', 'hand_landmarker_f16.task')

_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
    mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
    )
)

def extract_landmarks_from_image(image):
    """Extract 63-dimensional hand landmark vector from image."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    result = _landmarker.detect(mp_image)

    if result.hand_landmarks:
        landmarks = []
        for lm in result.hand_landmarks[0]:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks)  # shape: (63,)

    return None


def process_dataset():
    # streaming=True avoids writing raw image files to disk,
    # which fails on Windows due to colons in the dataset filenames.
    print("Loading dataset (streaming)...")
    dataset = load_dataset("dduka/guitar-chords", streaming=True)

    # Grab label names from the first sample's metadata
    first = next(iter(dataset['train']))
    # label is an int; get class names from features if available, else infer later
    label_names = None
    try:
        label_names = dataset['train'].features['label'].names
    except Exception:
        pass  # will store raw int labels and decode after

    landmarks_data = {
        'train': {'X': [], 'y': []},
        'test':  {'X': [], 'y': []},
    }

    for split in ['train', 'test']:
        print(f"\nProcessing {split} split...")
        skipped = 0

        for sample in tqdm(dataset[split], desc=split):
            image = np.array(sample['image'])
            raw_label = sample['label']
            label = label_names[raw_label] if label_names else raw_label

            lm = extract_landmarks_from_image(image)
            if lm is not None:
                landmarks_data[split]['X'].append(lm)
                landmarks_data[split]['y'].append(label)
            else:
                skipped += 1

        print(f"  Skipped (no hand detected): {skipped}")

    for split in ['train', 'test']:
        landmarks_data[split]['X'] = np.array(landmarks_data[split]['X'])
        landmarks_data[split]['y'] = np.array(landmarks_data[split]['y'])

    np.save('data/processed/landmarks_train.npy', landmarks_data['train'])
    np.save('data/processed/landmarks_test.npy',  landmarks_data['test'])

    print(f"\nSaved landmarks:")
    print(f"  Train: {len(landmarks_data['train']['X'])} samples")
    print(f"  Test:  {len(landmarks_data['test']['X'])} samples")
    print(f"  Feature shape: {landmarks_data['train']['X'].shape[1:]}")
    print(f"  Unique chords: {np.unique(landmarks_data['train']['y'])}")


if __name__ == "__main__":
    process_dataset()
