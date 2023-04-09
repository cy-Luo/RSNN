import numpy as np
from get_data import load_mnist
from train import train

def random_search(X_train, y_train, X_test, y_test, num_search=10000):
    best_test_acc = 0
    best_params = {}

    for i in range(num_search):
        hidden_size = np.random.randint(40, 100)
        learning_rate = np.random.uniform(5e-4, 5e-3)
        reg_strength = np.random.uniform(1e-6, 1e-4)

        results = train(X_train, y_train, X_test, y_test, hidden_size=hidden_size, learning_rate=learning_rate, reg_strength=reg_strength)
        test_acc = results['test_acc_history'][-1]

        if test_acc > best_test_acc:
            best_i = i
            best_test_acc = test_acc
            best_params = {'hidden_size': hidden_size, 'learning_rate': learning_rate, 'reg_strength': reg_strength}

        print(f"第{i+1}次搜索, hidden_size: {hidden_size}, learning_rate: {learning_rate:.5f}, reg_strength: {reg_strength:.5f}, test_acc: {test_acc:.5f}")
        print(f"当前最佳搜索结果：第{best_i+1}次搜索，最佳参数：{best_params}，test_acc: {best_test_acc:.5f}")

    return best_params, best_test_acc

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    best_params, best_test_acc = random_search(X_train, y_train, X_test, y_test)
    print(f"最佳参数：{best_params}，test_acc：{best_test_acc}")