import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os

import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

Categories = ['Infected', 'Healthy']

from google.colab import drive
drive.mount('/content/drive/')

flat_data_arr = []
target_arr = []

datadir = '/content/drive/MyDrive/IGTDU DATA'

for i in Categories:
    print(f'loading... category : {i}')
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (150, 150, 3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
df = pd.DataFrame(flat_data)
df['Target'] = target
df

# Splitting the data into train and test sets
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)
print('Splitted Successfully')

# SVM model
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(x_train, y_train)

# Making predictions
y_pred = classifier.predict(x_test)
y_pred

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(cm, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

# Model Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print('Accuracy  :', accuracy_score(y_test, y_pred) * 100, "%")
print('Precision :', precision_score(y_test, y_pred) * 100, "%")
print('Recall    :', recall_score(y_test, y_pred) * 100, "%")
print('F1 Score  :', f1_score(y_test, y_pred) * 100, "%")

# Plotting SVM model performance
import plotly.express as px

# Create a dataframe with the metrics and their values
df = pd.DataFrame({'Accuracy': [97.2067], 'Precision': [97.619], 'Recall': [96.4706], 'F1 Score': [97.0414]})

# Create a 3D scatter plot
fig = px.scatter_3d(df, x='Accuracy', y='Precision', z='Recall', color='F1 Score', size='F1 Score',
                    opacity=0.7, symbol='F1 Score', title='SVM Model Performance on Testing Dataset')

# Set axis labels and ranges
fig.update_layout(scene=dict(xaxis_title='Accuracy', yaxis_title='Precision', zaxis_title='Recall'),
                  scene_xaxis_range=[0, 100], scene_yaxis_range=[0, 100], scene_zaxis_range=[0, 100])

# Show the plot
fig.show()
