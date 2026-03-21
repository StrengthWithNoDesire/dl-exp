import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from utils import *
from model import Model, FullyConnectedLayer

def main():

    rd=42
    epochs = 150
    batch_size = 128
    learning_rate = 0.005
    optimizer = 'adam'
    l2_reg = True
    l2_lambda = 1e-3


    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"]


    # 对X进行归一化
    X = X / 255.0
    print("归一化完成")

    # one-hot编码
    y = np.eye(10)[y.astype(int)]  # 10个类别
    print("one-hot编码完成")



    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=rd)
    print("数据集划分完成")

    model = Model(learning_rate=learning_rate, optimizer=optimizer)
    model.add_layer(layer_input_dim=784, layer_output_dim=512, initialize='xavier', activation='leaky_relu', l2=l2_reg, l2_lambda=l2_lambda)
    model.add_layer(layer_input_dim=512, layer_output_dim=256, initialize='xavier', activation='leaky_relu', l2=l2_reg, l2_lambda=l2_lambda)
    model.add_layer(layer_input_dim=256, layer_output_dim=10, initialize='xavier', activation='linear', l2=l2_reg, l2_lambda=l2_lambda)

    print("模型构建完成, 开始训练")




    train_losses = []    
    val_losses = []
    train_accuracies = []    
    val_accuracies = []


    for epoch in range(epochs):
        m = X_train.shape[0]
        indices = np.random.permutation(m)

        X_train_shuffled = X_train.iloc[indices]
        y_train_shuffled = y_train[indices]

        epoch_train_loss = 0
        epoch_train_acc = 0
        num_batches = 0
        for i in range(0, m, batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]

            y_pred_logits = model.forward(X_batch)
            y_pred_probs = softmax(y_pred_logits)
            y_pred = np.argmax(y_pred_probs, axis=1)

            loss = cross_entropy_loss(y_batch, y_pred_probs)
            if l2_reg:
                l2_loss = sum(l2_regularization_loss(layer.W, l2_lambda) for layer in model.layers)
                loss += l2_loss
            accuracy = np.mean(y_pred == np.argmax(y_batch, axis=1))

            epoch_train_loss += loss 
            epoch_train_acc += accuracy
            num_batches += 1

            # print(f"Epoch {epoch + 1}, Batch {i // batch_size + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

            model.backward(y_pred_probs - y_batch)

        train_losses.append(epoch_train_loss / num_batches)
        train_accuracies.append(epoch_train_acc / num_batches)

        y_val_pred_logits = model.forward(X_val)
        y_val_pred_probs = softmax(y_val_pred_logits)
        y_val_pred = np.argmax(y_val_pred_probs, axis=1)
        y_val_label = np.argmax(y_val, axis=1)

        val_loss = cross_entropy_loss(y_val, y_val_pred_probs)
        if l2_reg:
            l2_loss = sum(l2_regularization_loss(layer.W, l2_lambda) for layer in model.layers)
            val_loss += l2_loss
        val_accuracy = np.mean(y_val_pred == y_val_label)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if epoch == epochs - 1:  # 最后一个epoch结束后绘制混淆矩阵
           plot_confusion_matrix(y_val_label, y_val_pred)
           plot_misclassified_samples(X_val, y_val_label, y_val_pred, num_samples=10, image_shape=(28, 28), random_seed=rd)

    plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, epochs)
    



if __name__ == "__main__":
    main()


