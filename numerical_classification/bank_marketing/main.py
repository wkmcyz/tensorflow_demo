import numpy as np
import tensorflow as tf
from keras.feature_column.dense_features import DenseFeatures
from scipy.io import arff
from tensorflow import feature_column
from keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd

# 加载数据
data = arff.loadarff(open('dataset.arff', 'r'))
dataframe = pd.DataFrame(data[0])  # 你的数据集的路径

# Create a DataFrame with NaN and inf values
# df = pd.DataFrame({'col1': [1.0, 2.0, np.nan, 4.0, 5.0], 'col2': [1.0, 2.0, 3.0, np.inf, 5.0]})
df = dataframe
# Check for NaN and inf values
print(df.isna().any())  # Check for NaN values
print(df.isin([np.inf, -np.inf]).any())  # Check for inf values

# Replace NaN and inf values
df = df.replace([np.inf, -np.inf], np.nan)  # Replace inf values with NaN
df = df.fillna(0)  # Replace NaN values with 0

# Convert to numpy array and make sure all data is float
numpy_array = df.to_numpy().astype(np.float32)

print("wahaha")

# 对于分类变量，我们需要将其转化为数值型变量。这可以通过one-hot编码或embedding等方式完成。
# 在这个例子中，我们使用one-hot编码。
# categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
# numeric_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
numeric_columns = ['V1', 'V6', 'V10', 'V12', 'V13', 'V14', 'V15']

feature_columns = []
# for header in categorical_columns:
#     vocabulary = dataframe[header].unique()
#     cat_col = feature_column.categorical_column_with_vocabulary_list(header, vocabulary)
#     one_hot = feature_column.indicator_column(cat_col)
#     feature_columns.append(one_hot)

for header in numeric_columns:
    numeric_col = feature_column.numeric_column(header)
    feature_columns.append(numeric_col)

# 现在我们已经定义了我们的特征列，我们可以构建我们的模型。
model = tf.keras.Sequential([
    DenseFeatures(feature_columns),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 我们将使用二元交叉熵作为损失函数，优化器使用Adam。
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 将数据集分割为训练集和测试集
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

# 训练模型
model.fit(train, epochs=10)

# 验证模型
loss, accuracy = model.evaluate(test)
print(f"loss : {loss}", "Accuracy", accuracy)
