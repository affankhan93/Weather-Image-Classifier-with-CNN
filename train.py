from model import CNN 
import torch
import torch.nn as nn
import torch.optim as optim
import argparse 
import time
from torchvision.transforms import transforms 
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score


def parser_arguments():
    parser = argparse.ArgumentParser()

    # All arguments.
    parser.add_argument('--data_dir', type=str, default=r"C:\Users\affan\OneDrive\Documents\Data Science\Weather Classification with CNN (Project)\dataset")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--out', type=str, default='weather_cnn.pth')

    return parser.parse_args()


def main():
    args = parser_arguments()

    # Check device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transform
    transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5] , std=[0.5, 0.5, 0.5])
])

    # Load dataset
    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    class_names = dataset.classes
    num_classes = len(class_names)
    print(f" Found {len(dataset)} images across {num_classes} classes: {class_names}")


    # Split into trian & test 
    train_size = int(0.8 * len(dataset)) # 80% for training 
    test_size = len(dataset) - train_size # 20% for testing 

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size]) 

    # Train & Test Dataloader

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False) 

    # Model CNN 
    model = CNN().to(DEVICE) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    # A schedular changes the learning rate during training instead of keeping it fixed.
    # without a schedular our optimizer uses the same learning rate (e.g 0.001) for every single batch across all epochs.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) 

    # Model training
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        start = time.time() 

        progress_bar = tqdm(train_loader, desc = f'Epoch {epoch+1}/{args.epochs}', unit = 'batch', leave =False)

        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        epoch_time = time.time() - start  # how many time it will take to complete one epochs it measure that. 

        test_acc, test_precision = evaluate(model, test_loader, DEVICE)

        print(f"Epoch {epoch+1}/{args.epochs} loss={train_loss:.4f} test_accuracy={test_acc * 100:.2f}% test_precision={test_precision:.4f} time={epoch_time:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.out)
            print(f"new best ({best_acc * 100:.2f}%), saved to {args.out}")

    print(f"\nTraining done. Best test accuracy: {best_acc * 100:.2f}%")
    print(f"Best model weights saved at: {args.out}")

# Evaluate
def evaluate(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average = 'weighted', zero_division = 0)
    return accuracy, precision


if __name__ == '__main__':
    main()