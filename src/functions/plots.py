import numpy as np
import matplotlib.pyplot as plt

from keras import models
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score, f1_score

def plot_class_distribution(y_train:np.ndarray, y_val:np.ndarray, y_test:np.ndarray):
    """
    Plots the distribution of classes in the training, validation, and test sets.
    :param y_train: Array of training labels.
    :param y_val: Array of validation labels.
    :param y_test: Array of test labels.
    :return: Dictionary of class weights.
    """
    train_counts = np.bincount(y_train)
    val_counts = np.bincount(y_val)
    test_counts = np.bincount(y_test)

    total_classes = len(train_counts)
    class_weights = {i: len(y_train) / (total_classes * count) for i, count in enumerate(train_counts)}
    total_weight = sum(class_weights.values())
    class_weights = {k: v / total_weight for k, v in class_weights.items()}

    plt.figure()
    x_labels = range(total_classes)
    plt.bar(x_labels, train_counts, width=0.2, label='Train', align='center')
    plt.bar([x + 0.25 for x in x_labels], val_counts, width=0.2, label='Validation', align='center')
    plt.bar([x + 0.5 for x in x_labels], test_counts, width=0.2, label='Test', align='center')
    plt.xticks(x_labels, [f"Class {i}" for i in x_labels])
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Number of Samples per Class in Each Dataset')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return total_classes, class_weights

def plot_history(history:list, metrics:list[str]):
    """
    Plots the training history of the model.
    :param history: List of dictionaries containing training and validation metrics.
    :param metrics: List of metrics to plot.
    """
    plt.figure(figsize=None)
    nrows = len(history)
    ncols = len(metrics)

    for i, h in enumerate(history):
        for j, key in enumerate(metrics):
            if key not in h:
                continue
            plt.subplot(nrows, ncols, i * ncols + j + 1)
            plt.plot(h[key], label=f'Training {key}')
            plt.plot(h[f'val_{key}'], label=f'Validation {key}')
            plt.title(f'Model {key}')
            plt.xlabel('Epoch')
            plt.ylabel(key)
            plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(datasets:list[tuple[str, np.ndarray, np.ndarray]], total_classes:int, figsize=None):
    """
    Plots confusion matrices for the given datasets.
    :param datasets: List of tuples containing dataset title, true labels, and predicted labels.
    :param figsize: Size of the figure.
    """
    _, axes = plt.subplots(1, len(datasets), figsize=figsize)
    for i, (title, y_true, y_pred) in enumerate(datasets):
        y_pred = np.argmax(y_pred, axis=1)
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            normalize='true',
            display_labels=[i for i in range(total_classes)],
            cmap=plt.cm.Blues,
            colorbar=False,
            ax=axes[i]
        )
        axes[i].set_title(f"{title}\n"
                          + f"Accuracy: {balanced_accuracy_score(y_true, y_pred):.2%}\n"
                          + f"F1-Score: {f1_score(y_true, y_pred):.2f}")

    plt.tight_layout()
    plt.show()


def plot_autoencoder_predictions(x_test:np.ndarray, x_test_encoded:np.ndarray, y_test:np.ndarray, y_test_pred:np.ndarray, decoder:models.Model):
    """
    Plots the original and predicted images from the autoencoder.
    :param x_test: Test images.
    :param y_test: Test labels.
    :param autoencoder: Trained autoencoder model.
    """

    # Choose N random images from the test set
    total = 10
    indices = np.random.choice(len(x_test), total, replace=False)

    orig_img = x_test[indices]
    orig_classes = y_test[indices]
    pred_img = decoder.predict(x_test_encoded[indices], verbose=0)
    pred_classes = y_test_pred[indices]

    # Plot the original and predicted images
    plt.figure(figsize=(15, 3.5))
    for i, _ in enumerate(indices):
        _ = plt.subplot(2, total, i + 1)
        plt.imshow(orig_img[i], cmap='gray')
        plt.title(f"Original: {orig_classes[i]}")
        plt.axis('off')

        pred = np.argmax(pred_classes[i])
        correct = pred == orig_classes[i]

        _ = plt.subplot(2, total, i + total + 1)
        plt.imshow(pred_img[i], cmap='gray')
        plt.title(f"Pred: {pred_classes[i][1]:.2f}", color='green' if correct else 'red')
        plt.axis('off')
    plt.show()