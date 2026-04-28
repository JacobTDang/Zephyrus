from datasets import load_dataset

# streaming=True avoids caching raw image files to disk,
# which fails on Windows due to colons in the filenames (e.g. "frame_2024-07-17 09:30:00.jpg")
dataset = load_dataset("dduka/guitar-chords", streaming=True)

# Print label names from the dataset info
info = load_dataset("dduka/guitar-chords", streaming=True)
for split, ds in info.items():
    ds_iter = iter(ds)
    sample = next(ds_iter)
    print(f"{split} — label type: {type(sample['label'])}, image size: {sample['image'].size}")
    break

print("Dataset accessible via streaming. Run extract_landmarks.py to process it.")
