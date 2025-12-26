import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from prettytable import PrettyTable
from torchvision import transforms, datasets

# مسیر ماژول MedMamba
sys.path.append("/kaggle/working/MedMamba-test")
from MedMamba import VSSM as medmamba


class ConfusionMatrix:
    """Confusion Matrix with accuracy, precision, recall, specificity and plotting."""

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, targets):
        for p, t in zip(preds, targets):
            self.matrix[p, t] += 1

    def summary(self):
        total_correct = sum(self.matrix[i, i] for i in range(self.num_classes))
        acc = total_correct / np.sum(self.matrix)
        print(f"Model accuracy: {acc:.4f}")

        table = PrettyTable()
        table.field_names = ["Class", "Precision", "Recall", "Specificity"]

        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            precision = round(TP / (TP + FP), 3) if TP + FP > 0 else 0.0
            recall = round(TP / (TP + FN), 3) if TP + FN > 0 else 0.0
            specificity = round(TN / (TN + FP), 3) if TN + FP > 0 else 0.0

            table.add_row([self.labels[i], precision, recall, specificity])

        print(table)

    def plot(self):
        plt.imshow(self.matrix, cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion Matrix')

        thresh = self.matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                value = int(self.matrix[y, x])
                plt.text(x, y, value,
                        ha='center', va='center',
                        color="white" if value > thresh else "black")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")  # ذخیره به فایل
        plt.close()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])}

    # Dataset
    train_dataset = datasets.ImageFolder(
        root="/kaggle/input/cpn-xray-dataset/CPN_Xray/train",
        transform=data_transform["train"])
    train_num = len(train_dataset)

    class_dict = dict((val, key) for key, val in train_dataset.class_to_idx.items())
    with open('class_indices.json', 'w') as f:
        json.dump(class_dict, f, indent=4)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(
        root="/kaggle/input/cpn-xray-dataset/CPN_Xray/val",
        transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nw)
    print(f"Training images: {train_num}, Validation images: {val_num}")

    # Model
    net = medmamba(num_classes=3)
    net.to(device)

    # Load pretrained weights
    model_weight_path = "/kaggle/working/MedMambaNet.pth"
    assert os.path.exists(model_weight_path), f"Cannot find {model_weight_path}"
    net.load_state_dict(torch.load(model_weight_path, map_location=device))

    # Labels
    labels = list(class_dict.values())
    confusion = ConfusionMatrix(num_classes=3, labels=labels)

    # Evaluation
    net.eval()
    with torch.no_grad():
        for val_images, val_labels in tqdm(validate_loader):
            outputs = net(val_images.to(device))
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            confusion.update(preds.cpu().numpy(), val_labels.cpu().numpy())

    # Results
    confusion.plot()
    confusion.summary()


if __name__ == '__main__':
    main()