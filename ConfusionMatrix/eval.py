import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")   # بک‌اند غیرگرافیکی
import matplotlib.pyplot as plt
from tqdm import tqdm
from prettytable import PrettyTable
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# مسیر ماژول MedMamba
sys.path.append("/kaggle/working/MedMamba-test")
from MedMamba import VSSM as medmamba


class ConfusionMatrix:
    """Confusion Matrix + Accuracy, Precision, Recall, F1"""

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.all_preds = []
        self.all_targets = []

    def update(self, preds, targets):
        for p, t in zip(preds, targets):
            self.matrix[p, t] += 1
        self.all_preds.extend(preds)
        self.all_targets.extend(targets)

    def summary(self):
        # Accuracy کلی
        acc = accuracy_score(self.all_targets, self.all_preds)
        print(f"\nOverall Accuracy: {acc:.4f}")

        # Precision, Recall, F1 برای هر کلاس
        precision = precision_score(self.all_targets, self.all_preds, average=None, zero_division=0)
        recall = recall_score(self.all_targets, self.all_preds, average=None, zero_division=0)
        f1 = f1_score(self.all_targets, self.all_preds, average=None, zero_division=0)

        table = PrettyTable()
        table.field_names = ["Class", "Precision", "Recall", "F1-Score"]
        for i, label in enumerate(self.labels):
            table.add_row([label,
                           round(precision[i], 3),
                           round(recall[i], 3),
                           round(f1[i], 3)])
        print(table)

        # Macro/Micro averages
        print(f"Macro Precision: {precision_score(self.all_targets, self.all_preds, average='macro'):.4f}")
        print(f"Macro Recall:    {recall_score(self.all_targets, self.all_preds, average='macro'):.4f}")
        print(f"Macro F1:        {f1_score(self.all_targets, self.all_preds, average='macro'):.4f}")
        print(f"Micro Precision: {precision_score(self.all_targets, self.all_preds, average='micro'):.4f}")
        print(f"Micro Recall:    {recall_score(self.all_targets, self.all_preds, average='micro'):.4f}")
        print(f"Micro F1:        {f1_score(self.all_targets, self.all_preds, average='micro'):.4f}")

    def plot(self, filename="confusion_matrix.png"):
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
        plt.savefig(filename)
        plt.close()
        print(f"Confusion matrix saved to {filename}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    data_transform = {
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])
    }

    # Validation dataset
    validate_dataset = datasets.ImageFolder(
        root="/kaggle/input/cpn-xray-dataset/CPN_Xray/test",
        transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=32,
                                                  shuffle=False,
                                                  num_workers=4)
    print(f"Validation images: {val_num}")

    # Model
    net = medmamba(num_classes=3)
    net.to(device)

    # Load pretrained weights
    model_weight_path = "/kaggle/working/MedMambaNet.pth"
    assert os.path.exists(model_weight_path), f"Cannot find {model_weight_path}"
    net.load_state_dict(torch.load(model_weight_path, map_location=device))

    # Labels
    class_dict = dict((val, key) for key, val in validate_dataset.class_to_idx.items())
    labels = list(class_dict.values())

    # Confusion Matrix
    confusion = ConfusionMatrix(num_classes=3, labels=labels)

    # Evaluation
    net.eval()
    with torch.no_grad():
        for val_images, val_labels in tqdm(validate_loader, desc="Validation"):
            outputs = net(val_images.to(device))
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            confusion.update(preds.cpu().numpy(), val_labels.cpu().numpy())

    # Results
    confusion.plot("confusion_matrix_val.png")
    confusion.summary()


if __name__ == '__main__':
    main()