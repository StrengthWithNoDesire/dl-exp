import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

def cross_entropy_loss(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    m = y_true.shape[0]
    epsilon = 1e-15
    true_class = np.argmax(y_true, axis=1)
    true_prob = y_pred[np.arange(m), true_class]
    loss = -np.mean(np.log(true_prob + epsilon))
    return loss





def l2_regularization_gradient(weights:np.ndarray, lambda_:float) -> np.ndarray:
    return lambda_ * weights

def l2_regularization_loss(weights:np.ndarray, lambda_:float) -> float:
    return (lambda_ / 2) * np.sum(np.square(weights))

def softmax(z:np.ndarray) -> np.ndarray:
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # for numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def sigmoid(z:np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

def relu(z:np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def leaky_relu(z:np.ndarray, alpha:float = 0.01) -> np.ndarray:
    return np.where(z > 0, z, alpha * z)

def leaky_relu_differential(z:np.ndarray, alpha:float = 0.01) -> np.ndarray:
    return np.where(z > 0, 1, alpha)

def tanh(z:np.ndarray) -> np.ndarray:
    return np.tanh(z)

def tanh_differential(z:np.ndarray) -> np.ndarray:
    t = tanh(z)
    return 1 - t**2

def sigmoid_differential(z:np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1 - s)

def relu_differential(z:np.ndarray) -> np.ndarray:
    return np.where(z > 0, 1, 0)


def plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, epochs=None):
    """
    绘制训练/验证损失和准确率曲线
    参数：
        train_losses: list, 每个epoch的训练损失
        val_losses: list, 每个epoch的验证损失
        train_accuracies: list, 每个epoch的训练准确率
        val_accuracies: list, 每个epoch的验证准确率
        epochs: int or None, 总epoch数，若为None则根据列表长度自动推断
    """
    if epochs is None:
        epochs = len(train_losses)
    x = range(1, epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x, train_losses, label='Train Loss', marker='o')
    plt.plot(x, val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(x, val_accuracies, label='Validation Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix', normalize=False):
    """
    绘制混淆矩阵
    参数：
        y_true: 真实标签，可以是one-hot编码或类别索引
        y_pred: 预测标签，格式同y_true
        labels: list, 类别名称（可选，若为None则显示数字标签）
        title: str, 图表标题
        normalize: bool, 是否归一化
    """
    # 转换为类别索引（如果是one-hot）
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    # 设置刻度标签：如果提供了labels则使用，否则使用True让seaborn自动生成数字标签
    xticklabels = labels if labels is not None else True
    yticklabels = labels if labels is not None else True

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', cbar=False,
                xticklabels=xticklabels, yticklabels=yticklabels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

def plot_misclassified_samples(X_test, y_true, y_pred, class_names=None, num_samples=10,
                                image_shape=None, random_seed=None):
    """
    可视化错误分类的样本
    参数：
        X_test: 测试数据，形状 (n_samples, ...) 可以是图像特征向量或原始图像
        y_true: 真实标签，one-hot或类别索引
        y_pred: 预测标签，one-hot或类别索引
        class_names: list, 类别名称（可选）
        num_samples: int, 最多显示的样本数
        image_shape: tuple, 如果X_test是扁平化图像，指定形状如 (28,28)；若为None且数据是2D则尝试直接显示
        random_seed: int, 随机种子，用于复现
    """
    # 转换为类别索引
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # 找出错误分类的索引
    error_indices = np.where(y_true != y_pred)[0]
    if len(error_indices) == 0:
        print("没有错误分类样本！")
        return

    if random_seed is not None:
        np.random.seed(random_seed)

    # 随机选择指定数量的错误样本
    num_samples = min(num_samples, len(error_indices))
    selected_indices = np.random.choice(error_indices, num_samples, replace=False)

    # 确定子图布局
    cols = 5
    rows = (num_samples + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))

    for i, idx in enumerate(selected_indices):
        sample = X_test[idx]

        # 处理图像数据：如果提供了image_shape且样本是一维，尝试重塑
        if image_shape is not None and sample.ndim == 1:
            try:
                sample = sample.reshape(image_shape)
            except:
                pass  # 若重塑失败则保持原样

        plt.subplot(rows, cols, i + 1)

        # 根据数据维度决定显示方式
        if sample.ndim == 2:  # 灰度图像
            plt.imshow(sample, cmap='gray')
            plt.axis('off')
        elif sample.ndim == 3 and sample.shape[-1] in [1, 3, 4]:  # 彩色图像（通道在最后）
            if sample.shape[-1] == 1:
                plt.imshow(sample.squeeze(), cmap='gray')
            else:
                plt.imshow(sample)
            plt.axis('off')
        else:
            # 非图像数据：显示特征的前几个数值
            feat_str = str(sample.flatten()[:5]) if sample.size > 0 else "[]"
            plt.text(0.5, 0.5, f'Features:\n{feat_str}', ha='center', va='center', fontsize=8)
            plt.axis('off')

        # 设置标题（真实 vs 预测）
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        if class_names is not None:
            true_label = class_names[true_label] if true_label < len(class_names) else true_label
            pred_label = class_names[pred_label] if pred_label < len(class_names) else pred_label
        plt.title(f'True: {true_label}\nPred: {pred_label}', fontsize=9)

    plt.suptitle('Misclassified Samples')
    plt.tight_layout()
    plt.show()