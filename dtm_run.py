# Import Modules Necessary
# Depending Upon your environment use PIP or PM.
# Or Install Dependencies via Requirements file
# View Readme.md for details

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib

# Loading Data
data = pd.read_csv("toys.csv")

# Converting Data
data['Color'] = data['Color'].astype('category').cat.codes
data['size'] = data['size'].astype('category').cat.codes
data['firmness'] = data['firmness'].astype('category').cat.codes
data['age'] = data['age'].astype('category').cat.codes

# Splitting Data (X) and labels (y)
X = data.drop('expensive', axis=1)
y = data['expensive']

# Split the data into training and testing sets. Specify as a standard 70% of the data to be used for training
# Here we are splitting the dataset into training and testing sets with a 70-30 ratio.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Save the split datasets to CSV files
# We save the split datasets into separate CSV files for future use and reference.
X_train.to_csv('./train/X_train_3.csv', index=False)
X_test.to_csv('./test/X_test_3.csv', index=False)
y_train.to_csv('./train/y_train_3.csv', index=False)
y_test.to_csv('./test/y_test_3.csv', index=False)

# Train a decision tree classifier model
# Here we train a decision tree classifier using the training data.
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Save the trained model to a file
# We save the trained decision tree model to a file for later use.
joblib.dump(clf, './model/dtm_model_3.pkl')

# Now for reference if the features change or after pre processing we can use the saved model for predictions.
# Load the trained model from the .pkl file.
# loaded_model = joblib.load('./model/dtm_model_2.pkl')

# Use the loaded model to make predictions on the new dataset
# y_pred_new = loaded_model.predict(X_new)

# Evaluate the performance of the trained decision tree classifier model using test data
# Here we evaluate the performance of the trained model using the test data.
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Save accuracy and confusion matrix to a text file
# We save the accuracy and confusion matrix to a text file for analysis and documentation.
with open('./results/results_2.txt', 'w') as f:
    f.write("Acc: {}\n".format(accuracy))
    f.write("Conf Mx:\n")
    f.write(str(conf_matrix))

# Plot the decision tree and save it as a PNG file
# Here we visualize the decision tree and save it as a PNG file for visualization purposes.
plt.figure(figsize=(12,8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['False', 'True'])
plt.savefig('./views/dtm_diagram_3.png')

plt.show()
