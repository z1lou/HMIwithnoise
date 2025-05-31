import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from keras.models import load_model
from keras.utils import to_categorical
from keras import initializers

# === CONFIGURATION ===
new_subject_data_dir = "new_subject/"
model_path = "model1.keras"
output_path = "subject_adapted_model.keras"

NUM_CLASSES = 19
SAMPLES_PER_CLASS = 2
ROWS_PER_SAMPLE = 100
NUM_FEATURES = 6

LR_BACKBONE = 1e-3
LR_HEAD = 1e-2
EPOCHS = 20
BATCH_SIZE = 2

# === LOAD FEW-SHOT DATA ===
def load_few_shot_data(folder_path, num_classes, samples_per_class, rows_per_sample, features):
    X, y = [], []
    for class_idx in range(1, num_classes + 1):
        file_path = os.path.join(folder_path, f'gesture{class_idx}.csv')
        df = pd.read_csv(file_path)
        for i in range(samples_per_class):
            start = i * rows_per_sample
            end = start + rows_per_sample
            if end <= len(df):
                sample = df.iloc[start:end, :features].values
                X.append(sample)
                y.append(class_idx - 1)
    X = np.array(X)
    y = to_categorical(np.array(y), num_classes)
    return X, y

X_train, y_train = load_few_shot_data(
    new_subject_data_dir,
    NUM_CLASSES,
    SAMPLES_PER_CLASS,
    ROWS_PER_SAMPLE,
    NUM_FEATURES
)

# === LOAD SAVED SCALER ===
scaler = joblib.load("scaler.pkl")  # Adjust path as needed

# Flatten, normalize, and reshape back
X_train_flat = X_train.reshape(-1, NUM_FEATURES)
X_train_scaled = scaler.transform(X_train_flat)
X_train = X_train_scaled.reshape(-1, ROWS_PER_SAMPLE, NUM_FEATURES)


# === LOAD BASE MODEL ===
model = load_model(model_path, compile=False)
model.trainable = True

# === Reinitialize final Dense layer (e.g., "dense_17") ===
def reinitialize_layer(model, layer_name):
    layer = model.get_layer(name=layer_name)
    init = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
    weights = layer.get_weights()
    new_weights = [init(w.shape) for w in weights]
    layer.set_weights(new_weights)

reinitialize_layer(model, "dense_17")  # Replace name if needed

# === MANUAL GRADIENT SCALING ===
optimizer = tf.keras.optimizers.Adam()

# Split trainable variables
backbone_vars = []
head_vars = []
for layer in model.layers[:-5]:  
    backbone_vars.extend(layer.trainable_variables)
for layer in model.layers[-5:]: 
    head_vars.extend(layer.trainable_variables)

# Convert to dataset
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(buffer_size=len(X_train)).batch(BATCH_SIZE)

loss_fn = tf.keras.losses.CategoricalCrossentropy()

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    epoch_loss = []
    for x_batch, y_batch in dataset:
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss = loss_fn(y_batch, logits)

        grads = tape.gradient(loss, backbone_vars + head_vars)
        grads_backbone = grads[:len(backbone_vars)]
        grads_head = grads[len(backbone_vars):]

        scaled_grads = []
        for g, v in zip(grads_backbone, backbone_vars):
            if g is not None:
                scaled_grads.append((g * LR_BACKBONE, v))
        for g, v in zip(grads_head, head_vars):
            if g is not None:
                scaled_grads.append((g * LR_HEAD, v))

        optimizer.apply_gradients(scaled_grads)
        epoch_loss.append(loss.numpy())

    print(f"Loss: {np.mean(epoch_loss):.4f}")

# === SAVE MODEL ===
model.save(output_path)
print(f"\n Adapted model saved to: {output_path}")
