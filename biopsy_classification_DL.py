import tensorflow as tf
from keras.layers import Dense
from keras import Sequential
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, matthews_corrcoef

# %% Preprocessing

data = pd.read_csv("D:/Datasets/Biopsy/biopsy.csv")
data.drop(data.columns[[26, 27]], inplace=True, axis=1)
data.replace('?', np.nan, inplace=True)
null_values = data.isna().sum()
null_values = null_values[null_values != 0]
impute_data = null_values
imputer = SimpleImputer()
data[impute_data.index] = imputer.fit_transform(data[impute_data.index])
data['Age'] = StandardScaler().fit_transform(data['Age'].to_numpy().reshape((-1, 1)))

# %% Splitting
data = np.asarray(data).astype(np.float32)
data[::, -1] = LabelEncoder().fit_transform(data[::, -1])
X, y = data[::, :-1], data[::, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

train_len = int(X_train.shape[0] * (80 / 100))
X_train, y_train, X_val, y_val = X_train[:train_len, ::], y_train[:train_len], X_train[train_len:, ::], y_train[
                                                                                                        train_len:]

# %% Model Building

model = Sequential([
    Dense(X.shape[1], activation='relu', name='input_layer'),
    Dense(1000, activation='relu', name='hidden_layer1'),
    Dense(2000, activation='relu', name='hidden_layer2'),
    Dense(3000, activation='relu', name='hidden_layer3'),
    Dense(2000, activation='relu', name='hidden_layer4'),
    Dense(1000, activation='relu', name='hidden_layer5'),
    Dense(1, activation=tf.keras.activations.sigmoid, name='output_layer')
])

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=80, shuffle=True, validation_data=(X_val, y_val), batch_size=20,
                    use_multiprocessing=True)

# %% Model Prediction
y_pred = np.round(model.predict(X_test))

# %% Plotting the Accuracy and Validation Accuracy Scores
val_accuracy, accuracy = history.history['val_accuracy'], history.history['accuracy']
confusion_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(confusion_mat, annot=True, fmt='d')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

# %%
plt.plot(accuracy)
plt.plot(val_accuracy)
plt.legend(['Accuracy', 'Validation Accuracy'])
plt.xlabel('Number of Epoch')
plt.ylabel('Score')
plt.show()

print(matthews_corrcoef(y_test, y_pred))

# %% Plotting the LOSS to identify the Overfitting

val_loss, fun_loss = history.history['val_loss'], history.history['loss']
plt.plot(val_accuracy)
plt.plot(val_loss)
plt.legend(['Validation Accuracy', 'Validation Loss'])
plt.xlabel('Number of Epoch')
plt.ylabel('Score')
plt.show()

# %% Average Metrics from history

print("Average Metrics")
print("------------------------------------------------------")
print("Loss -", np.round(np.average(history.history['loss']), 2))
print("Validation Loss -", np.round(np.average(history.history['val_loss']), 2))
print("Accuracy -", np.round(np.average(history.history['accuracy']), 2))
print("Validation Accuracy -", np.round(np.average(history.history['val_accuracy']), 2))
print("======================================================")
