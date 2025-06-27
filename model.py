# Importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
import joblib
from tensorflow import keras
from keras import Sequential


# Load dataset
df = pd.read_csv('diabetes.csv')

print(df.shape)

# Divide dataset into independent and dependent variables
X = df.drop(columns="Outcome")
y = df["Outcome"]

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Divide dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Build a neural network
model = tf.keras.Sequential([
     tf.keras.layers.Dense(16, activation="relu", input_shape=(X.shape[1],)),  # input layer
     tf.keras.layers.Dense(8, activation="relu"),  # hidden layer
     tf.keras.layers.Dense(1, activation="sigmoid")  # output layer
     
])

model.summary()

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test))

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32").flatten()

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

# Print metrics
print("The Accuracy is:", accuracy)
print("Precision is:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", cr)

# Save the model
model.save("Diabetes_model.h5")
print("Model has been saved.")
