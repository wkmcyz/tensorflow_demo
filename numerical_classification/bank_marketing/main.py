from typing import Tuple

import numpy as np
import tensorflow as tf
from keras.feature_column.dense_features_v2 import DenseFeatures
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow import feature_column, Tensor
from keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd

# 加载数据
data = arff.loadarff(open('dataset.arff', 'r'))
dataframe = pd.DataFrame(data[0], dtype=np.float)  # 你的数据集的路径
dataframe['Class'] = dataframe['Class'].astype(int)

# 对于分类变量，我们需要将其转化为数值型变量。这可以通过one-hot编码或embedding等方式完成。
# 在这个例子中，我们使用one-hot编码。
# categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
# numeric_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
numeric_columns = ['V1', 'V6', 'V10', 'V12', 'V13', 'V14', 'V15']

# 对类别标签进行编码
label_encoder = LabelEncoder()
dataframe['Class'] = label_encoder.fit_transform(dataframe['Class'])

# 对特征值进行归一化
# 对特征值进行归一化
scaler = MinMaxScaler()
dataframe[dataframe.columns[:-1]] = scaler.fit_transform(dataframe[dataframe.columns[:-1]])

# 将数据集分割为训练集和测试集
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)


def extract_x_y(df) -> Tuple:
    # 选取 DataFrame 的前 9 列
    df_selected = df.iloc[:, :7]

    # 将 DataFrame 转换成 Tensor
    x_tensor: Tensor = tf.convert_to_tensor(df_selected.values, dtype=tf.float32)

    y = df.iloc[:, 7]
    y_tensor = tf.convert_to_tensor(y.values, dtype=tf.float32)
    return x_tensor, y_tensor


train_x, train_y = extract_x_y(train)
test_x, test_y = extract_x_y(test)

feature_columns = []
for header in numeric_columns:
    numeric_col = feature_column.numeric_column(header)
    feature_columns.append(numeric_col)
# 现在我们已经定义了我们的特征列，我们可以构建我们的模型。
model = tf.keras.Sequential([
    # DenseFeatures(feature_columns),
    layers.Dense(128, input_dim=train_x.shape[1], activation='relu'),
    # layers.Dense(128, activation='relu', input_shape=(7,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    # layers.Dense(2, activation='softmax')
    layers.Dense(1, activation='sigmoid')
])

# 我们将使用二元交叉熵作为损失函数，优化器使用Adam。
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
# 训练模型
print("开始 fit ")
model.fit(train_x, train_y, validation_split=0.2, epochs=100, batch_size=10)

# 验证模型
loss, accuracy = model.evaluate(test_x, test_y)
print(f"loss : {loss}", "Accuracy", accuracy)
