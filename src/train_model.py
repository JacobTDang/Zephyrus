import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class ChordDataset(Dataset):
    def __init__(self, landmarks, labels):
        self.landmarks = torch.FloatTensor(landmarks)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        return self.landmarks[idx], self.labels[idx]


class CNN1D_ChordClassifier(nn.Module):
    """Input: (batch, 63) — 21 landmarks × (x, y, z)"""

    def __init__(self, num_chords):
        super().__init__()

        # Reshape to (batch, 3, 21): 3 channels (x,y,z), 21 landmark positions
        self.conv_layers = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_chords),
        )

    def forward(self, x):
        x = x.view(-1, 21, 3).transpose(1, 2)  # (batch, 3, 21)
        x = self.conv_layers(x)
        return self.classifier(x)


def train_model():
    train_data = np.load('data/processed/landmarks_train.npy', allow_pickle=True).item()

    le = LabelEncoder()
    y_all = le.fit_transform(train_data['y'])

    # Split 80/20 from the training landmarks (dataset test split is too small: 21 samples)
    X_train, X_test, y_train, y_test = train_test_split(
        train_data['X'], y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    num_chords = len(le.classes_)
    print(f"Chord classes ({num_chords}): {le.classes_}")
    print(f"Train: {len(X_train)}  Val: {len(X_test)}")

    train_loader = DataLoader(ChordDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader  = DataLoader(ChordDataset(X_test,  y_test),  batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model     = CNN1D_ChordClassifier(num_chords).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    best_acc = 0.0

    for epoch in range(100):
        model.train()
        correct = total = 0

        for landmarks, labels in train_loader:
            landmarks, labels = landmarks.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(landmarks), labels)
            loss.backward()
            optimizer.step()
            preds = model(landmarks).argmax(1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)

        train_acc = 100.0 * correct / total

        model.eval()
        t_correct = t_total = 0
        with torch.no_grad():
            for landmarks, labels in test_loader:
                landmarks, labels = landmarks.to(device), labels.to(device)
                t_correct += model(landmarks).argmax(1).eq(labels).sum().item()
                t_total   += labels.size(0)

        test_acc = 100.0 * t_correct / t_total
        scheduler.step(test_acc)

        print(f"Epoch {epoch+1:3d}/100 | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_chords':       num_chords,
                'label_encoder':    le,
                'test_acc':         test_acc,
            }, 'data/models/best_chord_model.pth')
            print(f"  -> Saved (best: {best_acc:.2f}%)")

    print(f"\nDone. Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    train_model()
