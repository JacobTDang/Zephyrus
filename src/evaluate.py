import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from train_model import CNN1D_ChordClassifier


def evaluate_model():
    checkpoint = torch.load('data/models/best_chord_model.pth', map_location='cpu')
    test_data  = np.load('data/processed/landmarks_test.npy', allow_pickle=True).item()

    le = checkpoint['label_encoder']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNN1D_ChordClassifier(checkpoint['num_chords']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    X = torch.FloatTensor(test_data['X']).to(device)
    y = le.transform(test_data['y'])

    with torch.no_grad():
        preds = model(X).argmax(1).cpu().numpy()

    print("\nClassification Report:")
    print(classification_report(y, preds, target_names=le.classes_))

    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('data/models/confusion_matrix.png')
    print("Confusion matrix saved to data/models/confusion_matrix.png")


if __name__ == "__main__":
    evaluate_model()
