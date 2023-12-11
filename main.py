import tarfile
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt


# Function to calculate classification accuracy
def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples
    return accuracy


# Function to unpickle data files
def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict





def load_train_test_batches(train_files, test_file):
    train_data_batches = []
    train_labels_batches = []

    for train_file in train_files:
        batch = unpickle(train_file)
        train_data_batch = np.array(batch[b'data'])
        train_labels_batch = np.array(batch[b'labels'])

        train_data_batches.append(train_data_batch)
        train_labels_batches.append(train_labels_batch)

    test_batch = unpickle(test_file)
    test_data = np.array(test_batch[b'data'])
    test_labels = np.array(test_batch[b'labels'])

    train_data = np.concatenate(train_data_batches, axis=0)
    train_labels = np.concatenate(train_labels_batches, axis=0)

    return train_data, train_labels, test_data, test_labels


# List of data files for training batches (data_batch_1 to data_batch_5) and test batch (test_batch)
train_files = ['cifar-10-batches-py/data_batch_{}'.format(i) for i in range(1, 6)]
test_file = 'cifar-10-batches-py/test_batch'

# Load data from training and test batches
X_train, y_train, X_test, y_test = load_train_test_batches(train_files, test_file)

# Convert NumPy arrays to PyTorch tensors
X_train, y_train, X_test, y_test = map(torch.tensor, (X_train, y_train, X_test, y_test))

# Reshape data to match the expected input shape of the CNN
X_train = X_train.reshape(-1, 3, 32, 32)
X_test = X_test.reshape(-1, 3, 32, 32)

# Normalize pixel values to the range [0, 1]
X_train = X_train.float() / 255.0
X_test = X_test.float() / 255.0


class CNN(nn.Module):
    def __init__(self, hidden_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def plot_examples(model, X_test, y_test, num_examples=5):
    random_indices = np.random.choice(len(X_test), num_examples, replace=False)

    plt.figure(figsize=(15, 3))

    for i, idx in enumerate(random_indices):
        image = X_test[idx].numpy().transpose(1, 2, 0)
        label_true = int(y_test[idx])

        with torch.no_grad():
            output = model(X_test[idx].unsqueeze(0))
            predicted_class = torch.argmax(output).item()

        plt.subplot(1, num_examples, i + 1)
        plt.imshow(image)
        plt.title(f'True: {label_true}, Predicted: {predicted_class}')
        plt.axis('off')

    plt.show()


def train_and_evaluate(hidden_sizes_list, learning_rates_list, num_epochs):
    report = ""

    for learning_rate in learning_rates_list:
        for hidden_size in hidden_sizes_list:
            improved_model = CNN(hidden_size)
            criterion = nn.CrossEntropyLoss()
            improved_optimizer = optim.Adam(improved_model.parameters(), lr=learning_rate)

            # Convert data to PyTorch DataLoader
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

            # Training loop with tqdm progress bar for the model
            start_time = time.time()
            for epoch in range(num_epochs):
                total_loss = 0.0
                correct_predictions = 0
                total_samples = 0
                loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
                for batch_idx, (batch_x, batch_y) in loop:
                    improved_optimizer.zero_grad()
                    outputs = improved_model(batch_x)
                    loss = criterion(outputs, batch_y.long())
                    loss.backward()
                    improved_optimizer.step()
                    total_loss += loss.item()

                    # Calculate training accuracy
                    _, predicted = torch.max(outputs, 1)
                    correct_predictions += torch.sum(predicted == batch_y).item()
                    total_samples += batch_y.size(0)

                    loop.set_postfix(loss=loss.item())

                training_accuracy = correct_predictions / total_samples
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Training Accuracy: {training_accuracy * 100:.2f}%, Average Loss: {total_loss / len(train_loader):.4f}")

            training_time = time.time() - start_time

            # Evaluate the model
            improved_model.eval()
            with torch.no_grad():
                outputs = improved_model(X_test)
                _, predicted = torch.max(outputs, 1)

            # Calculate and print the testing accuracy
            testing_accuracy = calculate_accuracy(y_test.numpy(), predicted.numpy())
            report += f"Hidden Size: {hidden_size}, Learning Rate: {learning_rate}\n"
            report += f"Accuracy: {testing_accuracy * 100:.2f}%\n"
            report += f"Training Time: {training_time:.2f} seconds\n\n"

            # Plot examples of correct and incorrect predictions
            plot_examples(improved_model, X_test, y_test)

    return report


hidden_sizes_list = [1024]
learning_rates_list = [0.001]
num_epochs = 10

report = train_and_evaluate(hidden_sizes_list, learning_rates_list, num_epochs)

with open("report.txt", "w") as report_file:
    report_file.write(report)

print("Report generated and saved as 'report.txt'")
