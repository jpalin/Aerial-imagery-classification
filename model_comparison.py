# Import TensorFlow and Keras utilities
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Visualization and numerical tools
import matplotlib.pyplot as plt
import numpy as np

# Scikit-learn for confusion matrix
from sklearn.metrics import confusion_matrix

# Custom modules for plotting and dataset handling
from cf_matrix import make_confusion_matrix
from aerial_dataset import dataset_creation, dataset_creation_custom
from roc_curves import plot_roc_curve


def revaluate_model(model_json_file, model_h5_file, val_ds):
    """
    Load a saved Keras model from JSON and H5 files, compile it,
    and evaluate it on the provided validation dataset.

    Args:
        model_json_file (str): Path to the model architecture in JSON format.
        model_h5_file (str): Path to the model weights in HDF5 format.
        val_ds (tf.data.Dataset): Validation dataset to evaluate the model.

    Returns:
        tf.keras.Model: The loaded and compiled model.
    """
    # Load model architecture from JSON file
    with open(model_json_file, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # Load weights into the model
    loaded_model.load_weights(model_h5_file)
    print("Loaded model from disk")

    # Compile the model before evaluation
    loaded_model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    # Evaluate model on validation set
    loss, accuracy = loaded_model.evaluate(val_ds, verbose=2)
    print(f"{loaded_model.metrics_names[1]}: {accuracy * 100:.2f}%")
    print(f"{loaded_model.metrics_names[0]}: {loss:.4f}")

    return loaded_model


def plot_confusion_matrix(val_ds, loaded_model, file_name1, file_name2):
    """
    Plots ROC curve and confusion matrix for a given model on a dataset.

    Args:
        val_ds (tf.data.Dataset): Dataset to generate predictions from.
        loaded_model (tf.keras.Model): Trained model to evaluate.
        file_name1 (str): File name for saving ROC curve (no extension).
        file_name2 (str): File name for saving confusion matrix (no extension).
    """
    y_pred = []
    y_true = []

    # Generate predictions and collect true labels
    for image_batch, label_batch in val_ds:
        y_true.append(label_batch)  # True one-hot labels
        preds = loaded_model.predict(image_batch)
        y_pred.append(np.argmax(preds, axis=-1))  # Predicted class indices

    # Stack all batches together
    correct_labels = tf.concat(y_true, axis=0)
    predicted_labels = tf.concat([tf.convert_to_tensor(p) for p in y_pred], axis=0)

    # Convert predicted labels to one-hot encoding
    encoded = to_categorical(predicted_labels, num_classes=3)
    predictions = tf.convert_to_tensor(encoded)

    # Convert labels to class indices
    y_true_indices = tf.math.argmax(correct_labels, axis=1)
    y_pred_indices = tf.math.argmax(predictions, axis=1)

    # Prepare data for ROC curve
    y_true_roc = np.array([correct_labels])
    y_pred_roc = np.array([encoded])

    # Plot ROC curves
    plot_roc_curve(np.squeeze(y_true_roc), np.squeeze(y_pred_roc), file_name1)

    # Generate and display confusion matrix
    cf_matrix = confusion_matrix(y_true_indices, y_pred_indices)
    labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    categories = ['Ocean', 'Ships', 'Unknown Floating Objects']

    make_confusion_matrix(
        cf_matrix,
        group_names=labels,
        categories=categories,
        figsize=(10, 6),
        cmap='binary'
    )

    # Save and show the figure
    plt.savefig(file_name2 + ".png", bbox_inches='tight')
    plt.show()


def main():
    """
    Main function to evaluate multiple saved models, generate their performance metrics,
    and visualize their confusion matrices and ROC curves.
    """
    # Load default validation dataset
    train_ds, val_ds = dataset_creation(224, 224, 8)

    # Load dataset for evaluating model trained with augmented set
    val_set_for_augmented = dataset_creation_custom(
        1.0, 224, 224, 8, "C:/eval_set_for_augmentation_exp_class_dist"
    )

    # Re-load and evaluate saved models
    baseline_model = revaluate_model("model_baseline.json", "model_baseline.h5", val_ds)
    tuned_model_one = revaluate_model("model_tuned_one.json", "model_tuned_one.h5", val_ds)
    tuned_model_two = revaluate_model("model_tuned_two.json", "model_tuned_two.h5", val_ds)
    tuned_model_two_augmented = revaluate_model(
        "model_tuned_two_augmented_set.json",
        "model_tuned_two_augmented_set.h5",
        val_set_for_augmented
    )

    # Generate confusion matrix and ROC curve for each model
    plot_confusion_matrix(val_ds, baseline_model, "ROC_baseline", "confusion_baseline")
    plot_confusion_matrix(val_ds, tuned_model_one, "ROC_tuned_one", "confusion_tuned_one")
    plot_confusion_matrix(val_ds, tuned_model_two, "ROC_tuned_two", "confusion_tuned_two")
    plot_confusion_matrix(val_set_for_augmented, tuned_model_two_augmented, "ROC_tuned_two_augmented", "confusion_tuned_two_augmented")


# Script entry point
if __name__ == '__main__':
    main()
