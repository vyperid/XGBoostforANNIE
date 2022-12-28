import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

allData = pd.read_csv()

X = allData.drop(["label"], axis=1)
y = allData["label"]

X.head()
y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

learning_rates = np.arange(0.01, 1, 0.05) #from 0.01 to 1 with the increase of 0.05

test_XG = []
train_XG = []

for learning_rate in learning_rates:
    
    xgb_classifier = xgb.XGBClassifier(eta=learning_rate) #initializing the xgboost classifier with the selected learning reate (eta) 
    
    xgb_classifier.fit(X_train, y_train) #fitting the model
    
    train_XG.append(xgb_classifier.score(X_train, y_train)) #getting the scores of the training samples

    test_XG.append(xgb_classifier.score(X_test, y_test)) #getting the scores of the testing samples
    

fig = plt.figure(figsize=(10,7))

plt.plot(learning_rates, train_XG, c='red', label='Train')
plt.plot(learning_rates, test_XG, c='m', label='Test')

plt.xlabel('Learning Rate')
plt.xticks(learning_rates)
plt.ylabel('Accuracy Score')

plt.ylim(0.5, 1) #limiting the y axis values

plt.legend(prop={'size' : 12}, loc=3)

plt.title('Accuracy socres vs Learning Rate of XGBoost for ANNIE data')

plt.show()
