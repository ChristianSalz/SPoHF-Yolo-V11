import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns

# Configuration
data_dir = "./Insect-Types-Classes"
img_size = (224, 224)
batch_size = 1
epochs = 30  
model_path = "./InsectClassificationModel/insect_classifier.keras"
metrics_dir = "./insect_classification_model_metrics"

# Create metrics directory
os.makedirs(metrics_dir, exist_ok=True)

# Data preparation with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.15,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size, subset="training", class_mode='binary', shuffle=True)
val_data = val_datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size, subset="validation", class_mode='binary', shuffle=False)

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*img_size, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print class indices to verify folder mapping
print("Class indices:", train_data.class_indices)
print(f"Total training samples: {train_data.samples}")
print(f"Total validation samples: {val_data.samples}")

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Train the model
history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=[early_stopping])

# Save the model
model.save(model_path)
print(f"Model saved to {model_path}")

# Print final accuracy
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

# ============== GENERATE METRICS ==============

print("\nGenerating metrics and visualizations...")

# 1. Plot training history
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(metrics_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved training history plot")

# 2. Get predictions on validation set
val_data.reset()
y_true = val_data.classes
y_pred_proba = model.predict(val_data, verbose=0).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

# 3. ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(metrics_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved ROC curve (AUC = {roc_auc:.4f})")

# 4. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
class_names = list(train_data.class_indices.keys())

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(metrics_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved confusion matrix")

# 5. Classification Report
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\nClassification Report:")
print(report)

# Save classification report to file
with open(os.path.join(metrics_dir, 'classification_report.txt'), 'w') as f:
    f.write("Classification Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(report)
    f.write("\n\n" + "=" * 50 + "\n")
    f.write(f"ROC AUC Score: {roc_auc:.4f}\n")
    f.write(f"Total Training Samples: {train_data.samples}\n")
    f.write(f"Total Validation Samples: {val_data.samples}\n")
    f.write(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}\n")
    f.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")

print(f"✓ Saved classification report")

# 6. Save training history as CSV
import pandas as pd
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(metrics_dir, 'training_history.csv'), index=False)
print(f"✓ Saved training history CSV")

print(f"\n✓ All metrics saved to: {metrics_dir}/")