import numpy as np
from train import predict
from get_data import load_mnist

def test(X_test, y_test):
    weights = np.load("model_weights.npz")
    W1, b1, W2, b2 = weights['W1'], weights['b1'], weights['W2'], weights['b2']

    y_pred = predict(X_test, W1, b1, W2, b2)
    test_acc = np.mean(y_pred == y_test)
    return test_acc

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    test_acc = test(X_test, y_test)
    print(f"Test accuracy: {test_acc * 100:.2f}%")