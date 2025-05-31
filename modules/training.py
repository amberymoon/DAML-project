import torch

def train_model(model, train_loader, val_loader, epochs=10):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for epoch in range(epochs):
        train_loss = 0
        correct = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            model.optimizer.zero_grad()
            outputs = model(inputs)
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
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = model.criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
        val_acc = correct / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")
