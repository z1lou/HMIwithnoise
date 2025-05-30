import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from keras.utils import to_categorical
from keras.optimizers import Adam
import random
from statistics import mean, stdev
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# === CONFIGURATION ===
train_data_dir = 'level3/'  
test_data_dir = 'level1/'    
num_classes = 19
repetitions_per_class = 20
rows_per_repetition = 100
features = 6
model_path = 'base_model1.keras'

# === FUNCTION TO LOAD DATA ===
def load_dataset(folder_path, num_classes, reps_per_class, rows_per_rep, features, label_offset=True):
    X, y = [], []
    for class_idx in range(1, num_classes + 1):
        file_path = os.path.join(folder_path, f'gesture{class_idx}.csv')
        df = pd.read_csv(file_path, header=None)
        data = df.values.reshape(reps_per_class, rows_per_rep, features)
        X.append(data)
        label = class_idx - 1 if label_offset else class_idx
        y += [label] * reps_per_class
    return np.vstack(X), np.array(y)


# === Metric Storage ===
num_runs = 10
accs = []
precisions = []
recalls = []
f1s = []
specificities = []


for run in range(num_runs):
    print(f"\n=== Run {run + 1}/{num_runs} ===")

    # --- Set different seed for each run ---
    seed = random.randint(0, 10000)

    # === Reload and Normalize Training Data ===
    X, y = load_dataset(train_data_dir, num_classes, repetitions_per_class, rows_per_repetition, features)
    X_flat = X.reshape(-1, features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat).reshape(-1, rows_per_repetition, features)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=seed)
    X_train = X_train.reshape(-1, rows_per_repetition, features)
    X_val = X_val.reshape(-1, rows_per_repetition, features)
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)

    # === Build and Train Model ===
    model = Sequential()
    model.add(Input(shape=(rows_per_repetition, features)))
    model.add(Conv1D(16, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=7, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat),
              epochs=50, batch_size=10, verbose=0)  # silence output

    # === Load and Normalize Test Set ===
    X_test, y_test = load_dataset(test_data_dir, num_classes, repetitions_per_class, rows_per_repetition, features)
    X_test_flat = X_test.reshape(-1, features)
    X_test_scaled = scaler.transform(X_test_flat).reshape(-1, rows_per_repetition, features)
    y_test_cat = to_categorical(y_test, num_classes)

    # === Evaluate Accuracy ===
    loss, acc = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
    accs.append(acc)

    # === Predict Labels ===
    y_pred_probs = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)

    # === Compute Metrics ===
    precisions.append(precision_score(y_true, y_pred, average='macro', zero_division=0))
    recalls.append(recall_score(y_true, y_pred, average='macro', zero_division=0))
    f1s.append(f1_score(y_true, y_pred, average='macro', zero_division=0))

    # === Specificity (manual per-class) ===
    cm = confusion_matrix(y_true, y_pred)
    spec_per_class = []
    for i in range(num_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        spec_per_class.append(specificity)
    specificities.append(np.mean(spec_per_class))

    print(f"Accuracy: {acc:.4f}, Precision: {precisions[-1]:.4f}, Recall: {recalls[-1]:.4f}, F1: {f1s[-1]:.4f}, Specificity: {specificities[-1]:.4f}")

# === SUMMARY ===
print("\n=== Summary over", num_runs, "runs ===")
print(f"Mean Accuracy: {mean(accs):.4f} ± {stdev(accs):.4f}")
print(f"Mean Precision: {mean(precisions):.4f} ± {stdev(precisions):.4f}")
print(f"Mean Recall: {mean(recalls):.4f} ± {stdev(recalls):.4f}")
print(f"Mean F1-score: {mean(f1s):.4f} ± {stdev(f1s):.4f}")
print(f"Mean Specificity: {mean(specificities):.4f} ± {stdev(specificities):.4f}")

