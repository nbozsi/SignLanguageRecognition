import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Set the data directory where your CSV files are located
data_dir = './landmark_data'  # Replace with your actual data directory
output_dir = './reports'
n_best_labels = 40

#Function definitions
def get_best_folders(data_dir, n_best=20):
    count_dict = {}

    for root,dirs,files in os.walk(data_dir):
        if len(dirs) == 0:
            label = root.split('/')[-1]
            count_dict[label] = len(files)

    sorted_count_dict = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)

    return [os.path.join(data_dir,label[0]) for label in sorted_count_dict[:n_best]]

def pad_sequence(seq, max_len):
    if len(seq) < max_len:
        pad_width = ((0, max_len - len(seq)), (0, 0))
        seq = np.pad(seq, pad_width, mode='constant')
    else:
        seq = seq[:max_len]
    return seq

def normalize_sequence(seq):
    mean = np.mean(seq, axis=0)
    std = np.std(seq, axis=0) + 1e-8
    seq = (seq - mean) / std
    return seq

def preprocess_sequence(seq, max_seq_length):
    seq = pad_sequence(seq, max_seq_length)
    seq = normalize_sequence(seq)
    return seq

# Get the most represented data
best_paths = get_best_folders('./landmark_data', n_best_labels)

print(f'Calculating with {len(best_paths)} labels')

# Collect file paths and labels
csv_files = []
labels = []

for path in best_paths:
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)) and f.endswith('.csv'):
            file_path = os.path.join(path, f)
            csv_files.append(file_path)
            # Extract label from filename (assuming format: label_something.csv)
            basename = os.path.basename(f)
            label_part = basename.split('_')[0]
            labels.append(label_part)

num_csv_files = len(csv_files)
print(f"Found {len(csv_files)} CSV files.")
print("Found labels:")
print(set(labels))
print("Number of found labels:")
print(len(set(labels)))

# Load data from CSV files into sequences
sequences = []
valid_labels = []  # To keep track of labels corresponding to valid sequences

for idx, file in enumerate(csv_files):
    try:
        df = pd.read_csv(file)
        if df.empty:
            print(f"Warning: File {file} contains no data. Skipping.")
            continue
        data = df.values.astype(np.float32)  # Shape: (num_frames, num_features)
        # Check for NaN or Inf and replace
        if np.isnan(data).any() or np.isinf(data).any():
            print(f"Sequence at index {len(sequences)} contains NaN or Inf values. Handling them.")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        sequences.append(data)
        valid_labels.append(labels[idx])  # Collect label names corresponding to valid sequences
    except Exception as e:
        print(f"Error reading {file}: {e}. Skipping.")

# Check if any sequences were loaded
if len(sequences) > 0:
    num_features_list = [seq.shape[1] for seq in sequences]
    unique_num_features = set(num_features_list)
    print(f"Unique number of features per sequence: {unique_num_features}")
    if len(unique_num_features) > 1:
        print("Error: Sequences have inconsistent number of features.")
        exit()
    else:
        num_features = num_features_list[0]
        print(f"All sequences have {num_features} features.")
else:
    print("No valid sequences found. Exiting.")
    exit()

# Implement feature engineering: compute velocities and augment features
for idx, data in enumerate(sequences):
    # Compute differences between consecutive frames
    velocities = np.diff(data, axis=0)
    # Pad the velocities to match the sequence length
    velocities = np.vstack([velocities, np.zeros((1, num_features))])
    # Concatenate original data with velocities
    augmented_data = np.concatenate([data, velocities], axis=1)
    sequences[idx] = augmented_data

# Update num_features
num_features = sequences[0].shape[1]

# Now, fit the LabelEncoder on valid_labels only
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(valid_labels)
num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")

# Replace valid_labels with labels_encoded
valid_labels = labels_encoded

# Optional: Handle class imbalance by selecting classes with sufficient samples
from collections import Counter

# Count the number of samples per class
class_counts = Counter(valid_labels)
print(f"Class counts before filtering: {class_counts}")

# Adjust the threshold to include all classes (set threshold to 1)
selected_classes = [cls for cls, count in class_counts.items() if count >= 1]

# Filter sequences and labels
filtered_sequences = []
filtered_labels = []

for seq, label in zip(sequences, valid_labels):
    if label in selected_classes:
        filtered_sequences.append(seq)
        filtered_labels.append(label)

# Update sequences and valid_labels
sequences = filtered_sequences
valid_labels = np.array(filtered_labels)

# Re-encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(valid_labels)
num_classes = len(label_encoder.classes_)
valid_labels = labels_encoded
print(f"Number of classes after filtering: {num_classes}")

# Proceed only if there are classes left
if num_classes == 0:
    print("No classes left after filtering. Exiting.")
    exit()

# Apply data augmentation
def augment_sequence(sequence):
    # Temporal augmentation: time stretching
    factor = np.random.uniform(0.8, 1.2)
    indices = np.round(np.arange(0, len(sequence), factor)).astype(int)
    indices = indices[indices < len(sequence)]
    sequence = sequence[indices]
    # Spatial augmentation: add noise
    noise = np.random.normal(0, 0.01, sequence.shape)
    sequence += noise
    return sequence

augmented_sequences = []
augmented_labels = []

for seq, label in zip(sequences, valid_labels):
    augmented_sequences.append(seq)
    augmented_labels.append(label)
    # Augment and add new sequence
    aug_seq = augment_sequence(seq.copy())
    augmented_sequences.append(aug_seq)
    augmented_labels.append(label)

# Update sequences and valid_labels
sequences = augmented_sequences
valid_labels = np.array(augmented_labels)

# Determine the maximum sequence length
max_seq_length = max([seq.shape[0] for seq in sequences])
print(f"Maximum sequence length: {max_seq_length}")

# Optionally, set a maximum sequence length limit
max_seq_length = min(max_seq_length, 100)  # Adjust as appropriate

# Pad sequences
padded_sequences = pad_sequences(
    sequences,
    maxlen=max_seq_length,
    dtype='float32',
    padding='post',   # Right-padding
    truncating='post' # Truncate sequences that are longer than max_seq_length
)

# Normalize data
# Reshape data to 2D for normalization
num_samples = padded_sequences.shape[0]
padded_sequences_2d = padded_sequences.reshape(-1, num_features)

# Normalize
scaler = StandardScaler()
padded_sequences_2d = scaler.fit_transform(padded_sequences_2d)

# Reshape back to 3D
padded_sequences = padded_sequences_2d.reshape(num_samples, max_seq_length, num_features)

# Split data into training and testing sets
seq_train, seq_test, labels_train, labels_test = train_test_split(
    padded_sequences, valid_labels, test_size=0.2, random_state=42, stratify=valid_labels
)

# Create TensorFlow datasets for training and testing
batch_size = 8  # Adjusted batch size due to limited data
train_dataset = tf.data.Dataset.from_tensor_slices((seq_train, labels_train)).shuffle(100).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((seq_test, labels_test)).batch(batch_size)

# Build the neural network model with enhanced architecture
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras.models import Model

input_seq = Input(shape=(max_seq_length, num_features))

x = Conv1D(filters=64, kernel_size=3, activation='relu')(input_seq)
x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
x = Dropout(0.5)(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dropout(0.5)(x)
x = GlobalAveragePooling1D()(x)
x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input_seq, outputs=output)

# Compile the model with adjusted optimizer and learning rate scheduler
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9
)

optimizer = optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

# Use categorical crossentropy since we have few classes
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# Callbacks for monitoring training
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Compute class weights based on valid_labels
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(valid_labels),
    y=valid_labels
)

class_weight_dict = dict(enumerate(class_weights))

# Train the model
history = model.fit(
    train_dataset,
    epochs=50,  # Adjust as needed
    validation_data=test_dataset,
    callbacks=[early_stopping],
    class_weight=class_weight_dict
)

# Collect predictions and true labels for confusion matrix
# Prepare test data
test_data = [preprocess_sequence(seq, max_seq_length) for seq in seq_test]
test_data = np.array(test_data)
# Get predictions
predictions = model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)

# Generate confusion matrix and classification report
cm = confusion_matrix(labels_test, predicted_labels)
# Convert class indices to class names (strings)
class_names = [str(label) for label in label_encoder.classes_]
cr = classification_report(labels_test, predicted_labels, target_names=class_names)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (n_best_labels={n_best_labels})')
plt.savefig(os.path.join(output_dir, f'confusion_matrix_{n_best_labels}.png'))
plt.close()

# Generate misclassification plot
misclassifications = np.array(labels_test) != np.array(predicted_labels)
misclassification_counts = np.zeros(num_classes)
total_counts = np.zeros(num_classes)

for i in range(len(labels_test)):
    true_label = labels_test[i]
    total_counts[true_label] += 1
    if misclassifications[i]:
        misclassification_counts[true_label] += 1

misclassification_rates = misclassification_counts / total_counts

plt.figure()
plt.bar(range(num_classes), misclassification_rates)
plt.xticks(range(num_classes), class_names, rotation=90)
plt.title(f'Misclassification Rates per Class (n_best_labels={n_best_labels})')
plt.xlabel('Class')
plt.ylabel('Misclassification Rate')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'misclassification_plot_{n_best_labels}.png'))
plt.close()

# Save classification report
with open(os.path.join(output_dir, f'classification_report_{n_best_labels}.txt'), 'w') as f:
    f.write(cr)

# Plot training & validation accuracy values
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title(f'Model accuracy (n_best_labels={n_best_labels})')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(os.path.join(output_dir, f'model_accuracy_{n_best_labels}.png'))
plt.close()

# Plot training & validation loss values
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(f'Model loss (n_best_labels={n_best_labels})')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(os.path.join(output_dir, f'model_loss_{n_best_labels}.png'))
plt.close()

# Save report including labels and number of CSV files used
report = ''
report += f'n_best_labels: {n_best_labels}\n'
report += f'Number of CSV files used: {num_csv_files}\n'
report += f'Number of sequences used: {len(sequences)}\n'
report += 'Labels used:\n'
for idx, label in enumerate(class_names):
    report += f'{idx}: {label}\n'
report += '\n'

with open(os.path.join(output_dir, f'report_{n_best_labels}.txt'), 'w') as f:
    f.write(report)

# Save the trained model
model.save(f'best_{n_best_labels}_report.h5')
print(f"Model has been saved as 'best_{n_best_labels}_report.h5'.")
