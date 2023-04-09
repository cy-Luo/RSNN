import numpy as np
import matplotlib.pyplot as plt
from get_data import load_mnist
import seaborn as sns


def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True)) #减去一个最大值
    return exps / np.sum(exps, axis=1, keepdims=True)

def compute_loss(y_true, y_pred, reg_strength, W1, W2):
    """计算损失函数，使用交叉熵损失函数"""
    num_samples = y_true.shape[0]
    loss = -np.sum(np.log(y_pred[np.arange(num_samples), y_true])) / num_samples
    loss += reg_strength * (np.sum(W1**2) + np.sum(W2**2)) / 2 #加入L2正则化
    return loss

def forward_pass(X, W1, b1, W2, b2):
    """前向传播函数（网络结构）"""
    h1 = relu(X.dot(W1) + b1) #隐藏层1
    h2 = h1.dot(W2) + b2 #隐藏层2
    y = softmax(h2) #输出层
    return y

def relu_derivative(Z):
    dZ = np.where(Z > 0, 1, 0)
    return dZ

def linear_derivative(X, W, dL_dY):
    dL_dW = np.dot(X.T, dL_dY)
    dL_db = np.sum(dL_dY, axis=0)
    return dL_dW, dL_db

def backpropagation(X, y, W1, b1, W2, b2, reg_strength, y_pred):
    """反向传播算法"""
    num_samples = X.shape[0]
    dL_dy = y_pred
    dL_dy[np.arange(num_samples), y] -= 1
    dL_dy /= num_samples

    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)
    dL_dW2, dL_db2 = linear_derivative(A1, W2, dL_dy)

    dL_dA1 = np.dot(dL_dy, W2.T)
    dZ1 = relu_derivative(Z1) * dL_dA1
    dL_dW1, dL_db1 = linear_derivative(X, W1, dZ1)


    dL_dW1 += reg_strength * W1
    dL_dW2 += reg_strength * W2

    return dL_dW1, dL_db1, dL_dW2, dL_db2


def sgd(W, b, dW, db, learning_rate):
    """梯度下降"""
    W -= learning_rate * dW
    b -= learning_rate * db
    return W, b

def train(X_train, y_train, X_test, y_test, hidden_size=50, learning_rate=1e-3, reg_strength=1e-5, num_iters=100, batch_size=128, learning_rate_decay=0.95):
    input_size = X_train.shape[1]
    output_size = np.max(y_train) + 1

    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros(output_size)

    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []

    num_train = X_train.shape[0]

    for i in range(num_iters):
        batch_indices = np.random.choice(num_train, batch_size, replace=True)
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        y_train_pred = forward_pass(X_batch, W1, b1, W2, b2)
        y_test_pred = forward_pass(X_test, W1, b1, W2, b2)

        train_loss = compute_loss(y_batch, y_train_pred, reg_strength, W1, W2)
        train_loss_history.append(train_loss)
        test_loss = compute_loss(y_test, y_test_pred, reg_strength, W1, W2)
        test_loss_history.append(test_loss)

        dW1, db1, dW2, db2 = backpropagation(X_batch, y_batch, W1, b1, W2, b2, reg_strength, y_train_pred)

        W1, b1 = sgd(W1, b1, dW1, db1, learning_rate)
        W2, b2 = sgd(W2, b2, dW2, db2, learning_rate)

        if i % 5 == 0:
            train_acc = (predict(X_train, W1, b1, W2, b2) == y_train).mean()
            test_acc = (predict(X_test, W1, b1, W2, b2) == y_test).mean()
            train_acc_history.append(train_acc)
            test_acc_history.append(test_acc)
            print(f"iteration {i+1}/{num_iters}: loss {train_loss}, test_acc {test_acc}")

        learning_rate *= learning_rate_decay

    return {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
        'train_loss_history': train_loss_history,
        'test_loss_history': test_loss_history,
        'train_acc_history': train_acc_history,
        'test_acc_history': test_acc_history
    }

def predict(X, W1, b1, W2, b2):
    h1 = relu(X.dot(W1) + b1)
    scores = h1.dot(W2) + b2
    return np.argmax(scores, axis=1)

def visualize_weights(W1, W2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(W1.ravel(), bins=50)
    ax1.set_title('Layer 1 Weights')
    ax1.set_xlabel('Weight')
    ax1.set_ylabel('Frequency')

    ax2.hist(W2.ravel(), bins=50)
    ax2.set_title('Layer 2 Weights')
    ax2.set_xlabel('Weight')
    #ax2.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def visualize_weights_grid(W1, ncols=10):
    nrows = W1.shape[1] // ncols
    image_size = int(np.sqrt(W1.shape[0]))

    grid = np.zeros((nrows * (image_size + 1), ncols * (image_size + 1)))

    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            weight_image = W1[:, idx].reshape(image_size, image_size)
            weight_image = (weight_image - np.min(weight_image)) / (np.max(weight_image) - np.min(weight_image))
            grid[i * (image_size + 1):(i + 1) * (image_size + 1) - 1, j * (image_size + 1):(j + 1) * (image_size + 1) - 1] = weight_image

    plt.imshow(grid, cmap='gray')
    plt.axis('off')
    plt.title('Layer 1 Weights')
    plt.show()

def visualize_weights_heatmap(W2):
    plt.figure(figsize=(8, 6))
    sns.heatmap(W2, cmap="crest", center=0, annot=False, cbar=True, xticklabels=False, yticklabels=False)
    plt.title('Layer 2 Weights')
    plt.show()

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()

    hidden_size=92
    learning_rate=0.004796531598292601
    reg_strength=7.042926575877955e-05
    num_iters=1000
    batch_size=64
    learning_rate_decay=0.95

    results = train(X_train, y_train, X_test, y_test, hidden_size, learning_rate, reg_strength, num_iters, batch_size, learning_rate_decay)

    W1, b1, W2, b2 = results['W1'], results['b1'], results['W2'], results['b2']
    np.savez("model_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)

    visualize_weights(W1, W2)
    visualize_weights_grid(W1)
    visualize_weights_heatmap(W2)

    plt.plot(results['train_loss_history'], label='train')
    plt.plot(results['test_loss_history'], label='test')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

    plt.plot(results['train_acc_history'], label='train')
    plt.plot(results['test_acc_history'], label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.show()

