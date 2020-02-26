import time as t
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as T

from model import Net


def train(model, train_loader, criterion, optimizer, device):

    running_loss = 0.0
    running_acc = 0.0
    total = 0.0

    model.train()

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        out = model(x)

        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        running_acc += torch.sum(out.max(1)[1] == y).item()
        total += float(y.size(0))

    return running_loss / total, running_acc / total


def test(model, test_loader, criterion, device):

    running_loss = 0.0
    running_acc = 0.0
    total = 0.0

    model.eval()

    for x, y in test_loader:

        with torch.no_grad():

            x = x.to(device)
            y = y.to(device)

            out = model(x)

            loss = criterion(out, y)
            running_loss += loss.item() * x.size(0)
            running_acc += torch.sum(out.max(1)[1] == y).item()
            total += float(y.size(0))

    return running_loss / total, running_acc / total


def plot(losses, accuracies):

    plt.title("model loss")
    plt.plot(losses["train"], label="train")
    plt.plot(losses["test"], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    plt.title("model accuracy")
    plt.plot(accuracies["train"], label="train")
    plt.plot(accuracies["test"], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = T.Resize((224, 224))
    to_tensor = T.ToTensor()
    train_transforms = T.Compose(
        [
            resize,
            # T.Pad((0, 0.3), padding_mode="reflect"),
            T.RandomHorizontalFlip(0.5),
            to_tensor,
            normalize,
        ]
    )
    test_transforms = T.Compose([resize, to_tensor, normalize])

    train_set = torchvision.datasets.ImageFolder(
        "data/MIT_split/train", transform=train_transforms
    )
    test_set = torchvision.datasets.ImageFolder(
        "data/MIT_split/test", transform=test_transforms
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    accuracies = {"test": [], "train": []}
    losses = {"test": [], "train": []}

    for epoch in range(150):

        start = t.time()
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        accuracies["train"].append(train_acc)
        losses["train"].append(train_loss)
        print("-" * 74)
        print(
            "| End of epoch: {:3d} | Time: {:.2f}s | Train loss: {:.3f} | Train acc: {:.3f}|".format(
                epoch + 1, t.time() - start, train_loss, train_acc
            )
        )

        start = t.time()
        val_loss, val_acc = test(model, test_loader, criterion, device)
        accuracies["test"].append(val_acc)
        losses["test"].append(val_loss)
        print("-" * 74)
        print(
            "| End of epoch: {:3d} | Time: {:.2f}s | Val loss: {:.3f} | Val acc: {:.3f}|".format(
                epoch + 1, t.time() - start, val_loss, val_acc
            )
        )
    plot(losses, accuracies)


if __name__ == "__main__":
    main()
