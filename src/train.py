import torch
import os
import json

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, 
                save_best_model_path, save_historial_path):
    
    historial = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_precision_acc = 0.0
    os.makedirs(os.path.dirname(save_best_model_path), exist_ok=True)

    for epoch in range(num_epochs):
        # Training
        # Encendemos el modo entrenamiento del modelo para activar el dropout
        model.train()
        
        train_running_loss = 0.0
        train_correctos = 0
        train_total = 0

        for inputs, labels in train_loader:
            # Reseteamos los gradientes del optimizador antes de cada paso de entrenamiento
            optimizer.zero_grad()
            # Obtenemos las predicciones del modelo
            outputs = model(inputs)
            # Calculamos la perdida
            loss = criterion(outputs, labels)
            # Calculamos los gradientes de la perdida en base a los parametros del modelo
            loss.backward()
            # Actualizamos los parametros del modelo usando el optimizador
            optimizer.step()
            #Acumulamos la perdida
            train_running_loss += loss.item() * inputs.size(0)

            # Obtenemos las predicciones del modelo y calculamos la cantidad de correctos
            _, predicciones = torch.max(outputs, 1)
            train_correctos += torch.sum(predicciones == labels.data).item()
            train_total += labels.size(0)
        
        epoch_train_loss = train_running_loss / len(train_loader.dataset)
        epoch_train_acc = train_correctos / train_total

        # Validation
        # Encendemos el modo evaluación del modelo para desactivar el dropout
        model.eval()

        val_running_loss = 0.0
        val_correctos = 0
        val_total = 0

        # Desactivamos el calculo de gradientes para ahorrar memoria
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Obtenemos las predicciones del modelo
                outputs = model(inputs)
                # Calculamos la perdida
                loss = criterion(outputs, labels)
                # Acumulamos la perdida de validacion
                val_running_loss += loss.item() * inputs.size(0)
                
                # Obtenemos las predicciones del modelo y calculamos la cantidad de correctos
                _, predicciones = torch.max(outputs, 1)
                val_correctos += torch.sum(predicciones == labels.data).item()
                val_total += labels.size(0)

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correctos / val_total

        print(f"Época [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} {'**' if epoch_val_acc > best_precision_acc else ''}")

        # Guardamos los resultados de la epoca en el historial
        historial['train_loss'].append(epoch_train_loss)
        historial['train_acc'].append(epoch_train_acc)
        historial['val_loss'].append(epoch_val_loss)
        historial['val_acc'].append(epoch_val_acc)

        # Guardamos el modelo si la precision de validacion es mejor que la mejor hasta ahora
        if epoch_val_acc > best_precision_acc:
            best_precision_acc = epoch_val_acc
            torch.save(model.state_dict(), save_best_model_path)
            print(f'Modelo guardado en: {save_best_model_path}')

    with open(save_historial_path, 'w') as f:
        json.dump(historial, f)

    return historial