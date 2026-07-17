import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = r"C:\Projects\sleeping-monitor\labeled_zone_dataset\train"
TEST_DIR = r"C:\Projects\sleeping-monitor\labeled_zone_dataset\test"
MODEL_OUT_PATH = r"C:\Projects\sleeping-monitor\zone_model_cnn.pt"

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        # Input size: 1x264x264 (grayscale) -> resize to 1x64x64 for speed
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) # 32x32
        x = self.pool(self.relu(self.conv2(x))) # 16x16
        x = self.pool(self.relu(self.conv3(x))) # 8x8
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    classes = train_dataset.classes
    print(f"Classes found: {classes}")
    
    model = SimpleCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\nTraining CNN for 5 epochs...")
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/5, Loss: {running_loss/len(train_loader):.4f}")
        
    print("\nEvaluating CNN on Test Set...")
    model.eval()
    all_preds = []
    all_labels = []
    all_probas = []
    softmax = nn.Softmax(dim=1)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probas = softmax(outputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
            
    print(classification_report(all_labels, all_preds, target_names=classes))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    print("\nSample Probability Output (First 5 test frames):")
    for i in range(5):
        print(f"True: {classes[all_labels[i]]:<15} Pred: {classes[all_preds[i]]:<15}")
        for j, class_name in enumerate(classes):
            print(f"  P({class_name}): {all_probas[i][j]*100:.1f}%")
            
    torch.save(model.state_dict(), MODEL_OUT_PATH)
    print(f"\nModel saved to {MODEL_OUT_PATH}")

if __name__ == "__main__":
    main()
