import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
from sklearn.metrics import accuracy_score
import model as md
import tensorflow as tf
import matplotlib
from LIBWO import LIBWO
from sklearn.metrics import precision_score, recall_score, f1_score

matplotlib.use('AGG')

np.set_printoptions(threshold=np.inf)

version = 'v3'


def create_train_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    return x_train, y_train, x_test, y_test


def resnet18_model_predict(resnet18, model_name, it_acc, it_val_acc, it_loss, it_val_loss, val_labels):
    # Get various evaluation result data
    score = resnet18.evaluate(val_data, val_labels, verbose=0)
    val_labels = [int(element) for element in val_labels]
    test_loss = 'Test Loss : {:.4f}'.format(score[0])
    test_accuracy = 'Test Accuracy : {:.4f}'.format(score[1])
    predicted_classes = np.argmax(resnet18.predict(val_data), axis=1)
    acc_score = accuracy_score(val_labels, predicted_classes)

    # Calculate precision, recall, and F1 score
    precision = precision_score(val_labels, predicted_classes, average='macro')
    recall = recall_score(val_labels, predicted_classes, average='macro')
    f1 = f1_score(val_labels, predicted_classes, average='macro')

    # Print the evaluation results
    print('Model Name:', model_name)
    print('Training Loss:', it_loss)
    print('Test Loss:', test_loss)
    print('Training Accuracy:', it_acc)
    print('Test Accuracy:', test_accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1)

if __name__ == '__main__':
    # Generate training and test datasets
    train_data, train_label, val_data, val_label = create_train_data()
    # Model parameters
    model_param = {
        "test_data": val_data,
        "test_label": val_label,
        "data": train_data,
        "label": train_label
    }

    # Lagrange Black Widow Algorithm parameters
    libwo_param = {
        "pop": 3,  # Population size
        "MaxIter": 3,  # Maximum iterations
        "dim": 1,  # Dimensionality
        "lb": -0.0001 * np.ones([2, 1]),  # Lower bound [dim, 1]
        "ub": 0.1 * np.ones([2, 1]),  # Upper bound [dim, 1]
    }

    # Training model with default learning rate
    # default_model = md.resnet18_model()
    # cnn_default_model = default_model.model_create(0.01)
    # default_history = cnn_default_model.fit(train_data, train_label, epochs=1, batch_size=8, validation_split=0.1)
    # default_accuracy = default_history.history['accuracy']
    # default_val_accuracy = default_history.history['val_accuracy']
    # default_loss = default_history.history['loss']
    # default_val_loss = default_history.history['val_loss']
    # resnet18_model_predict(cnn_default_model, 'default',
    #                   default_accuracy, default_val_accuracy, default_loss, default_val_loss, val_label)


    # BWO = BWO(model_param, bwo_param)
    # best_err, best_learn_rate = BWO.BWO()
    # print("Best accuracy after Black Widow optimization:", best_err)
    # print("Best initial learning rate after Black Widow optimization:", best_learn_rate)
    # bwo_model = md.resnet18_model()
    # resnet18_bwo_model = bwo_model.model_create(best_learn_rate)
    # bwo_history = resnet18_bwo_model.fit(train_data, train_label, epochs=30, batch_size=128, validation_split=0.1)
    # bwo_accuracy = bwo_history.history['accuracy']
    # bwo_val_accuracy = bwo_history.history['val_accuracy']
    # bwo_loss = bwo_history.history['loss']
    # bwo_val_loss = bwo_history.history['val_loss']
    # resnet18_model_predict(resnet18_bwo_model, 'pdo',
    #                        bwo_accuracy, bwo_val_accuracy, bwo_loss, bwo_val_loss, val_label)

    LIBWO = LIBWO(model_param, libwo_param)
    best_err, best_learn_rate = LIBWO.LIBWO()
    print("Best accuracy after Black Widow optimization:", 1 - best_err)
    print("Best initial learning rate after Black Widow optimization:", best_learn_rate)
    bwo_model = md.resnet18_model()
    resnet18_bwo_model = bwo_model.model_create(best_learn_rate)
    bwo_history = resnet18_bwo_model.fit(train_data, train_label, epochs=40, batch_size=8, validation_split=0.1)
    bwo_accuracy = bwo_history.history['accuracy']
    bwo_val_accuracy = bwo_history.history['val_accuracy']
    bwo_loss = bwo_history.history['loss']
    bwo_val_loss = bwo_history.history['val_loss']
    resnet18_model_predict(resnet18_bwo_model, 'libwo',
                           bwo_accuracy, bwo_val_accuracy, bwo_loss, bwo_val_loss, val_label)
