import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import optuna

#Set up data paths
train_dir = "train"
test_dir = "test"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Data transformations (Adjust image size for ResNet if needed)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # For ResNet compatibility
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomCrop(224, padding=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ResNet
])

#Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

#Define the ResNet model
def create_model(dropout_rate=0.5):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Use pretrained ResNet50
    num_features = model.fc.in_features

    # Modify the fully connected layers for binary classification
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )

    return model

#Define objective function for Bayesian optimization
def objective(trial):
    #hyperparameters
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)

    #Initialize model
    model = create_model(dropout_rate=dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    #Train the model for 3 epochs for tuning
    for epoch in range(3):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    #Evaluate on test 
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    return test_loss / len(test_loader)

#Bayesian optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

#Best hyperparameters
print("Best hyperparameters:", study.best_params)

#Train the model with best hyperparameters
best_params = study.best_params
model = create_model(dropout_rate=best_params['dropout_rate']).to(device)
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
criterion = nn.BCELoss()

epochs = 10
train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []

for epoch in range(epochs):
    model.train()
    train_loss, correct_train, total_train = 0, 0, 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(correct_train / total_train)

    #Evaluate on the test set
    model.eval()
    test_loss, correct_test, total_test = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)

    test_losses.append(test_loss / len(test_loader))
    test_accuracies.append(correct_test / total_test)
    print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}")

#Generate metrics and save plots
all_preds = [int(p[0]) for p in all_preds]
all_labels = [int(l[0]) for l in all_labels]

print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

plt.figure()
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Testing Loss')
plt.savefig('training_testing_loss.jpeg')

plt.figure()
plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Testing Accuracy')
plt.savefig('training_testing_accuracy.jpeg')

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 7))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.jpeg')

#Save model weights
torch.save(model.state_dict(), 'resnet_model_weights.pth')
