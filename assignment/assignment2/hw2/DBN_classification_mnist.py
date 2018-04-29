import numpy as np
import pickle

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import confusion_matrix
from dbn.tensorflow import SupervisedDBNClassification


# Loading dataset
from utils import load_mnist
(x_train, y_train), (x_test, y_test) = load_mnist()
x_train_1 = np.reshape(x_train,(-1,28*28))
#x_train_1 = x_train_1[:100]
y_train_1 = np.where(y_train==1)[1]
#y_train_1 = y_train_1[:100]
x_test_1 = np.reshape(x_test,(-1,28*28))
#x_test_1 = x_test_1[:10];
y_test_1 = np.where(y_test==1)[1]
#y_test_1 = y_test_1[:10];

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(x_train_1, y_train_1)

# pickle.dump( classifier, open( "DBN_classifier.pkl", "wb" ) )

# Test
Y_pred = np.array(classifier.predict(x_test_1))
print('Done.\nAccuracy: %f' % accuracy_score(y_test_1, Y_pred))

acc_1nn_1 = sum(Y_pred == y_test_1)/float(len(y_test_1))
print("Test accuracy of Deep Belief Nets: %.3f"%acc_1nn_1)
print(confusion_matrix(y_test_1, Y_pred, labels=range(10)))