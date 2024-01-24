import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# Load the saved embeddings and labels from the pickle file
input_pickle_file =  np.load("face_embeddings_done.npz")



embeddings = input_pickle_file['arr_0']
labels = input_pickle_file['arr_1']

print(embeddings.shape)
print(labels.shape)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Train a classifier (SVM as an example)
model = SVC(kernel='linear',probability=True)
model.fit(X_train,y_train)

# Predict labels for the test set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
