import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from cf_matrix import make_confusion_matrix
from aerial_dataset import dataset_creation, dataset_creation_custom
from roc_curves import plot_roc_curve


def revaluate_model(model_json_file, model_h5_file, val_ds):
    # load json and create model
    json_file = open(model_json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_h5_file)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    loss, score = loaded_model.evaluate(val_ds, verbose=2)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score * 100))
    print(loaded_model.metrics_names[0], "{:10.4f}".format(loss))
    # print("{:10.4f}".format(x))
    return loaded_model


def plot_confusion_matrix(val_ds, loaded_model, file_name1, file_name2):
    y_pred = []  # store predicted labels
    y_true = []  # store true labels
    # iterate over the dataset
    for image_batch, label_batch in val_ds:  # use dataset.unbatch() with repeat
        # append true labels
        y_true.append(label_batch)
        # compute predictions
        preds = loaded_model.predict(image_batch)
        # append predicted labels
        y_pred.append(np.argmax(preds, axis=- 1))

    # convert the true and predicted labels into tensors
    correct_labels = tf.concat([item for item in y_true], axis=0)
    predicted_labels = tf.concat([item for item in y_pred], axis=0)
    # one hot encode
    encoded = to_categorical(predicted_labels, num_classes=3)
    # convert to tensors
    predictions = tf.convert_to_tensor(encoded)
    # convert to value array
    y_true = tf.math.argmax(correct_labels, 1)
    y_pred = tf.math.argmax(predictions, 1)
    y_true_roc = np.array([correct_labels])
    y_pred_roc = np.array([encoded])
    # roc curves plotting
    plot_roc_curve(np.squeeze(y_true_roc), np.squeeze(y_pred_roc), file_name1)
    # create confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    categories = ['Ocean', 'Ships', 'Unknown Floating Objects']
    make_confusion_matrix(cf_matrix,
                          group_names=labels,
                          categories=categories,
                          figsize=(10, 6),
                          cmap='binary')
    # save to file and show
    plt.savefig(file_name2 + ".png", bbox_inches='tight')
    plt.show()

def main():
    train_ds, val_ds = dataset_creation(224,224,8)
    val_set_for_augmented = dataset_creation_custom(1.0, 224, 224, 8, "C:/eval_set_for_augmentation_exp_class_dist")
    #train_set_for_augmented = dataset_creation_custom(1.0, 224, 224, 8, "C:/aerial_images_with_added_augmented_set_class_dist")
    # revaluate all models
    baseline_model = revaluate_model("model_baseline.json", "model_baseline.h5", val_ds)
    tuned_model_one = revaluate_model("model_tuned_one.json", "model_tuned_one.h5", val_ds)
    tuned_model_two = revaluate_model("model_tuned_two.json", "model_tuned_two.h5", val_ds)
    tuned_model_two_augmented_set = revaluate_model("model_tuned_two_augmented_set.json",
                                                   "model_tuned_two_augmented_set.h5",
                                                   val_set_for_augmented)
    # plot confusion matrixes for all models
    plot_confusion_matrix(val_ds, baseline_model, "ROC_baseline", "confusion_baseline.png")
    plot_confusion_matrix(val_ds, tuned_model_one, "ROC_tuned_one", "confusion_tuned_one.png")
    plot_confusion_matrix(val_ds, tuned_model_two, "ROC_tuned_two", "confusion_tuned_two.png")
    plot_confusion_matrix(val_set_for_augmented, tuned_model_two_augmented_set, "ROC_tuned_two_augmented", "confusion_tuned_two_augmented_set.png")

if __name__ == '__main__':
    main()

