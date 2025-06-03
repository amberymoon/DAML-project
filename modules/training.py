import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

def train_model(model, train_loader, val_loader, epochs=100, showfig=True):

    losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        train_loss = 0
        correct = 0

        for inputs, labels in train_loader:
            model.optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.to(torch.int64)
            loss = model.criterion(outputs, labels)
            loss.backward()
            model.optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

        train_acc = correct / len(train_loader.dataset)
        model.scheduler.step()

        # Valutazione
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                labels = labels.to(torch.int64)
                loss = model.criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
        val_acc = correct / len(val_loader.dataset)

        losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")
    
    if showfig:
        show_classifier_history(train_accuracies, val_accuracies, losses)


def show_classifier_history(train_acc, val_acc, train_loss):
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({'font.size': 14})
    plt.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(train_loss, label='Loss')
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

    fig2,  ax2 = plt.subplots(1, 1, figsize=(8, 4))
    ax2.plot(train_acc, label='Train Accuracy')
    ax2.plot(val_acc, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    print("avg of last 5 val acc:", np.mean(val_acc[-5:]))


def train_regression_model(model, train_loader, val_loader, epochs=100, showfig=True):
    train_mse = []
    val_mse = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        train_loss = 0
        correct = 0

        for inputs, labels in train_loader:
            model.optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            assert outputs.dtype == labels.dtype == torch.float, "Output and labels must have the same dtype"
            loss = model.criterion(outputs, labels)
            loss.backward()
            model.optimizer.step()

            # print(torch.stack((outputs, outputs.round(), labels), dim=1))
            new_correct = (outputs.round() == labels).sum().item()
            correct += new_correct
            # print(new_correct, "/", len(labels), "correct in this batch")

            train_loss += loss.item()

        train_acc = correct / (len(train_loader.dataset))

        train_accuracies.append(train_acc)

        train_mse.append(train_loss / len(train_loader))
        model.scheduler.step()

        # Valutazione
        val_loss = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs).squeeze()
                loss = model.criterion(outputs, labels)
                val_loss += loss.item()
                val_acc = (outputs.round() == labels).sum().item() / len(labels)

        val_accuracies.append(val_acc)

        val_mse.append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}/{epochs} "
              f"| Train Loss: {train_loss:.2f} "
              f"| Val Loss: {val_loss:.2f} "
              f"| Train Acc: {train_acc:.3f} "
              f"| Val Acc: {val_acc:.3f}")

    if showfig:
        show_regressor_history(train_mse, val_mse, train_accuracies, val_accuracies)


def show_regressor_history(train_mse, val_mse, train_accuracies, val_accuracies):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.rcParams.update({'font.size': 14})
    ax.plot(train_mse, label='Train MSE')
    ax.plot(val_mse, label='Validation MSE')
    ax.set_title('Training and Validation MSE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    plt.tight_layout()


#valutazione finale
def evaluate_metrics(model, loader, model_type='classification'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            if model_type == 'classification':
                preds = outputs.argmax(dim=1)
            elif model_type == 'regression':
                preds = outputs.squeeze()
                preds = outputs.round().long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    print("\nTest Evaluation:")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print("\nDetailed per-class metrics:")
    print(classification_report(all_labels, all_preds))