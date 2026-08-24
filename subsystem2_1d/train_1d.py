# subsystem2_1d/train_1d.py
import os
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Import the dataloader factory from our local module
from subsystem2_1d.dataloader_1d import get_1d_dataloaders
# Ensure your model builder is imported here. For example:
from subsystem2_1d.classifiers_1d import build_model 

def train_model():
    # Configuration
    epochs = 50
    patience = 5
    batch_size = 1024
    data_dir = "data"  # Path relative to the project root
    checkpoint_dir = "subsystem2_1d/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    # 2. Fetch the real dataloaders
    train_loader, val_loader, _ = get_1d_dataloaders(data_dir=data_dir, batch_size=batch_size, num_workers=2)

    # Initialize model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Assuming an 11-class classification output (for RadioML 2016.10a)
    model = build_model("cnn1d", num_classes=11).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 4. Early Stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # 3. Main Training Loop for 50 epochs
    for epoch in range(1, epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * signals.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        avg_train_loss = train_loss / train_total
        train_accuracy = 100.0 * train_correct / train_total

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                
                outputs = model(signals)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * signals.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        avg_val_loss = val_loss / val_total
        val_accuracy = 100.0 * val_correct / val_total

        # 6. Print real-time epoch statistics
        print(f"Epoch [{epoch}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} - Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} - Val Acc: {val_accuracy:.2f}%")

        # 4 & 5. Early Stopping and Checkpoint Saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save best model checkpoint
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Best model saved to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[Early Stopping] Validation loss didn't improve for {patience} epochs. Stopping early!")
                break

if __name__ == "__main__":
    train_model()
