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
                                 (0.5, 0.5, 0.5))]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])
    }

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

    validate_dataset = datasets.ImageFolder(
        root="/kaggle/input/cpn-xray-dataset/CPN_Xray/val",
        transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nw)

    test_dataset = datasets.ImageFolder(
        root="/kaggle/input/cpn-xray-dataset/CPN_Xray/test",
        transform=data_transform["test"])
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=nw)

    print(f"Training images: {train_num}, Validation images: {val_num}, Test images: {test_num}")

    # Model
    net = medmamba(num_classes=3)
    net.to(device)

    # Load pretrained weights
    model_weight_path = "/kaggle/input/medmamba1/MedMambaNet.pth"
    assert os.path.exists(model_weight_path), f"Cannot find {model_weight_path}"
    net.load_state_dict(torch.load(model_weight_path, map_location=device))

    # Labels
    labels = list(class_dict.values())

    # ---------- Validation ----------
    confusion_val = ConfusionMatrix(num_classes=3, labels=labels)
    net.eval()
    with torch.no_grad():
        for val_images, val_labels in tqdm(validate_loader, desc="Validation"):
            outputs = net(val_images.to(device))
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            confusion_val.update(preds.cpu().numpy(), val_labels.cpu().numpy())

    confusion_val.plot()
    confusion_val.summary()

    # ---------- Test ----------
    confusion_test = ConfusionMatrix(num_classes=3, labels=labels)
    with torch.no_grad():
        for test_images, test_labels in tqdm(test_loader, desc="Test"):
            outputs = net(test_images.to(device))
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            confusion_test.update(preds.cpu().numpy(), test_labels.cpu().numpy())

    confusion_test.plot()
    confusion_test.summary()