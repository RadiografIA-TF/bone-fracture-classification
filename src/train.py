import torch
import os
import json
from tqdm import tqdm
from sklearn.metrics import recall_score


def train_model(model, train_loader, val_loader, criterion, optimizer,
                 num_epochs, save_best_model_path, save_historial_path,
                 device,
                 metric='acc',
                 scheduler=None,
                 patience=None,
                 use_amp=True,     # <- reincorporado, default True ya que tu GPU lo necesita
                 desc_prefix='Epoch'):
    os.makedirs(os.path.dirname(save_best_model_path), exist_ok=True)

    historial = {
        'train_loss': [], 'train_acc': [], 'train_recall': [],
        'val_loss': [], 'val_acc': [], 'val_recall': []
    }

    best_metric_value = 0.0
    early_stop_counter = 0

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == 'cuda'))

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_train_preds, all_train_labels = [], []

        train_bar = tqdm(train_loader, desc=f'{desc_prefix} {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        epoch_train_recall = recall_score(all_train_labels, all_train_preds)

        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_val_recall = recall_score(all_val_labels, all_val_preds)

        historial['train_loss'].append(epoch_train_loss)
        historial['train_acc'].append(epoch_train_acc)
        historial['train_recall'].append(epoch_train_recall)
        historial['val_loss'].append(epoch_val_loss)
        historial['val_acc'].append(epoch_val_acc)
        historial['val_recall'].append(epoch_val_recall)

        print(f'Época [{epoch+1}/{num_epochs}] | '
              f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} Recall: {epoch_train_recall:.4f} | '
              f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} Recall: {epoch_val_recall:.4f}')

        if scheduler is not None:
            val_metric_for_scheduler = epoch_val_acc if metric == 'acc' else epoch_val_recall
            scheduler.step(val_metric_for_scheduler)

        current_metric_value = epoch_val_acc if metric == 'acc' else epoch_val_recall

        if current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            torch.save(model.state_dict(), save_best_model_path)
            print(f'[*] Nuevo mejor modelo guardado ({metric}: {current_metric_value:.4f}) en: {save_best_model_path}')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if patience is not None and early_stop_counter >= patience:
                print(f'Early stopping por falta de mejora en {metric}.')
                break

    with open(save_historial_path, 'w') as f:
        json.dump(historial, f)

    return historial