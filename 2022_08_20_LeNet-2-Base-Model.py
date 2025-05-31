import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical


# === CONFIGURATION ===
train_data_dir = 'train_dataset/'  
test_data_dir = 'test_dataset/'    
num_classes = 19
repetitions_per_class = 20
rows_per_repetition = 100
features = 6
num_runs = 15

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

# === METRIC STORAGE ===
accs = []
precisions = []
recalls = []
f1s = []
specificities = []

class_metrics = {
    "precision": {str(i): [] for i in range(num_classes)},
    "recall": {str(i): [] for i in range(num_classes)},
    "f1": {str(i): [] for i in range(num_classes)},
    "specificity": {str(i): [] for i in range(num_classes)},
}

# === MAIN LOOP ===
for run in range(num_runs):
    print(f"\n=== Run {run + 1}/{num_runs} ===")
    seed = random.randint(0, 10000)
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Load and normalize training data
    X, y = load_dataset(train_data_dir, num_classes, repetitions_per_class, rows_per_repetition, features)
    X_flat = X.reshape(-1, features)
    scaler = StandardScaler()
    X_scaled_flat = scaler.fit_transform(X_flat)
    joblib.dump(scaler, "scaler.pkl")  # Save the scaler
    X_scaled = X_scaled_flat.reshape(-1, rows_per_repetition, features)


    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=seed)
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)

    # Build model
    model = Sequential()
    model.add(tf.keras.Input(shape=(rows_per_repetition, features)))
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

    model.compile(optimizer=Adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat),
              epochs=50, batch_size=10, verbose=0)

    # Load and normalize test data
    X_test, y_test = load_dataset(test_data_dir, num_classes, repetitions_per_class, rows_per_repetition, features)
    X_test_scaled = scaler.transform(X_test.reshape(-1, features)).reshape(-1, rows_per_repetition, features)
    y_test_cat = to_categorical(y_test, num_classes)

    # Evaluate
    loss, acc = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
    accs.append(acc)

    y_pred_probs = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)

    precisions.append(precision_score(y_true, y_pred, average='macro', zero_division=0))
    recalls.append(recall_score(y_true, y_pred, average='macro', zero_division=0))
    f1s.append(f1_score(y_true, y_pred, average='macro', zero_division=0))

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    print(f"\nPer-class metrics (Run {run + 1}):")
    for label in report:
        if label.isdigit():
            i = int(label)
            prec = report[label]["precision"]
            rec = report[label]["recall"]
            f1s_ = report[label]["f1-score"]
            class_metrics["precision"][label].append(prec)
            class_metrics["recall"][label].append(rec)
            class_metrics["f1"][label].append(f1s_)
            print(f"Class {i+1}: Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1s_:.4f}")

    # Specificity
    cm = confusion_matrix(y_true, y_pred)
    for i in range(num_classes):
        label = str(i)
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) != 0 else 0
        class_metrics["specificity"][label].append(spec)
        print(f"Class {i+1}: Specificity: {spec:.4f}")
    specificities.append(np.mean([class_metrics["specificity"][str(i)][-1] for i in range(num_classes)]))

# === Save to Excel ===
with pd.ExcelWriter("per_class_metrics.xlsx") as writer:
    for metric_name, class_dict in class_metrics.items():
        df = pd.DataFrame(class_dict)
        df.index = [f"Run_{i+1}" for i in range(num_runs)]
        df.to_excel(writer, sheet_name=metric_name)
        print(f"Saved '{metric_name}' to Excel.")

# === Summary ===
from statistics import mean, stdev
print("\n=== Summary over", num_runs, "runs ===")
print(f"Mean Accuracy: {mean(accs):.4f} ± {stdev(accs):.4f}")
print(f"Mean Precision: {mean(precisions):.4f} ± {stdev(precisions):.4f}")
print(f"Mean Recall: {mean(recalls):.4f} ± {stdev(recalls):.4f}")
print(f"Mean F1-score: {mean(f1s):.4f} ± {stdev(f1s):.4f}")
print(f"Mean Specificity: {mean(specificities):.4f} ± {stdev(specificities):.4f}")
