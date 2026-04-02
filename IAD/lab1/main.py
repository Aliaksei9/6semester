import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

CSV_PATH = 'Health_Risk_Dataset.csv'
OUT_DIR = 'classification_outputs'
RANDOM_STATE = 42
TEST_SIZE = 0.2
IMPUTER_NEIGHBORS = 10
KNN_NEIGHBORS = 5
LOGISTIC_MAX_ITER = 2000
EPOCHS = 55
BATCH_SIZE = 32
LEARNING_RATE = 0.001
QUANTILE_LOW = 0.25
QUANTILE_HIGH = 0.75
IQR_MULTIPLIER = 1.5
FIGURE_WIDTH = 10
FIGURE_HEIGHT = 5
DECIMAL_PLACES = 4

os.makedirs(OUT_DIR, exist_ok=True)


def data_preprocessing(df)->object:
    df_filled = df.copy()

    df_filled['Risk_Level'] = df_filled['Risk_Level'].astype('category')
    df_filled['On_Oxygen'] = df_filled['On_Oxygen'].astype('category')
    df_filled['Consciousness'] = df_filled['Consciousness'].astype('category')

    numeric_cols = df_filled.select_dtypes(include=['float64', 'int64']).columns

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_filled[numeric_cols])

    imputer = KNNImputer(n_neighbors=IMPUTER_NEIGHBORS)
    imputed_data = imputer.fit_transform(scaled_data)

    df_filled[numeric_cols] = imputed_data

    categories_consciousness = [["A", "P", "C", "V", "U"]]
    encoder_consciousness = OrdinalEncoder(categories=categories_consciousness)
    df_filled["Consciousness"] = encoder_consciousness.fit_transform(df_filled[["Consciousness"]])

    # Risk_Level — Ordinal Encoding
    categories_risk = [["Low", "Normal", "Medium", "High"]]
    encoder_risk = OrdinalEncoder(categories=categories_risk)
    df_filled["Risk_Level"] = encoder_risk.fit_transform(df_filled[["Risk_Level"]])


    feature = np.array(df_filled[["On_Oxygen"]])

    one_hot = OneHotEncoder(sparse_output=False)
    encoded_feature = one_hot.fit_transform(feature)

    new_cols = one_hot.get_feature_names_out(["On_Oxygen"])

    encoded_df = pd.DataFrame(encoded_feature, columns=new_cols, index=df_filled.index)

    df_filled = pd.concat([df_filled.drop(columns=["On_Oxygen"]), encoded_df], axis=1)

    Q1 = df_filled['Heart_Rate'].quantile(QUANTILE_LOW)
    Q3 = df_filled['Heart_Rate'].quantile(QUANTILE_HIGH)
    IQR = Q3 - Q1
    lower_bound = Q1 - IQR_MULTIPLIER * IQR
    upper_bound = Q3 + IQR_MULTIPLIER * IQR

    df_filled = df_filled[(df_filled['Heart_Rate'] >= lower_bound) & (df_filled['Heart_Rate'] <= upper_bound)]
    return df_filled


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return accuracy, f1


def decision_tree_model(X_train, X_test, y_train, y_test, metrics_list):
    name = "Decision Tree"
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc, f1 = compute_metrics(y_test, y_pred)
    metrics_list.append([name, acc, f1])
    return model


def knn_model(X_train, X_test, y_train, y_test, metrics_list):
    name = f"KNN (k={KNN_NEIGHBORS})"
    model = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc, f1 = compute_metrics(y_test, y_pred)
    metrics_list.append([name, acc, f1])

    return model


def logistic_regression_model(X_train, X_test, y_train, y_test, metrics_list):
    name = "Logistic Regression"
    model = LogisticRegression(max_iter=LOGISTIC_MAX_ITER, multi_class='multinomial', solver='lbfgs', random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc, f1 = compute_metrics(y_test, y_pred)
    metrics_list.append([name, acc, f1])


    return model


def gaussian_nb_model(X_train, X_test, y_train, y_test, metrics_list):
    name = "Gaussian NB"
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc, f1 = compute_metrics(y_test, y_pred)
    metrics_list.append([name, acc, f1])

    return model


class HealthRiskNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(HealthRiskNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        #self.relu = nn.Sigmoid()
        #self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def print_epoch_info(epoch, train_loss, train_acc, test_loss, test_acc):
    print(f"Epoch {epoch + 1}: Train Loss {train_loss:.{DECIMAL_PLACES}f}, Train Acc {train_acc:.{DECIMAL_PLACES}f}, Test Loss {test_loss:.{DECIMAL_PLACES}f}, Test Acc {test_acc:.{DECIMAL_PLACES}f}")


def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, out_dir):
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.savefig(os.path.join(out_dir, 'nn_loss.png'))
    plt.close()

    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.savefig(os.path.join(out_dir, 'nn_accuracy.png'))
    plt.close()


def neural_network_model(X_train, X_test, y_train, y_test, metrics_list, out_dir=OUT_DIR):
    name = "Neural Network"

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    model = HealthRiskNN(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor).item()
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_correct = (test_predicted == y_test_tensor).sum().item()
            test_acc = test_correct / len(y_test_tensor)

        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print_epoch_info(epoch, train_loss, train_acc, test_loss, test_acc)

    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, out_dir)

    y_pred = test_predicted.numpy()
    acc, f1 = compute_metrics(y_test, y_pred)
    metrics_list.append([name, acc, f1])

    return model


def compare_classification_models(X_train, X_test, y_train, y_test, out_dir=OUT_DIR):
    metrics_list = []
    models = {}
    models['Decision Tree'] = decision_tree_model(X_train, X_test, y_train, y_test, metrics_list)
    models['KNN'] = knn_model(X_train, X_test, y_train, y_test, metrics_list)
    models['Logistic Regression'] = logistic_regression_model(X_train, X_test, y_train, y_test, metrics_list)
    models['Gaussian NB'] = gaussian_nb_model(X_train, X_test, y_train, y_test, metrics_list)
    models['Neural Network'] = neural_network_model(X_train, X_test, y_train, y_test, metrics_list, out_dir=out_dir)

    df_results = pd.DataFrame(metrics_list, columns=['Model', 'Accuracy', 'F1_macro'])
    df_results = df_results.set_index('Model').round(DECIMAL_PLACES)
    print("\nСравнительная таблица метрик:\n")
    print(df_results.to_string())
    return df_results, models


def main():
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV файл не найден по пути: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print("Исходный размер датасета:", df.shape)

    df_proc = data_preprocessing(df)
    print("Размер после предобработки:", df_proc.shape)

    df_proc = df_proc.drop(columns=['Patient_ID'])

    X = df_proc.drop(columns=['Risk_Level'])
    y = df_proc['Risk_Level'].values


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    print("Train:", X_train.shape, "Test:", X_test.shape)
    df_results, trained_models = compare_classification_models(X_train, X_test, y_train, y_test, out_dir=OUT_DIR)


if __name__ == '__main__':
    main()