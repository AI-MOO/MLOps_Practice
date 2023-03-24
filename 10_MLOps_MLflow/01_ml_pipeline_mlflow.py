import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn 

training_data = pd.read_csv('../00_data/storepurchasedata.csv')
training_data.head()

# mlflow configuration 
mlflow.tracking.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment(experiment_name="KNN_Classifier")

with mlflow.start_run(run_name = "second_run") as run:

    # spliting the dataset into training features and target 
    X = training_data.iloc[:, :-1].values
    y = training_data.iloc[:,-1].values

    # splitting the dataset into train\test 80%:20% subsets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.60,random_state=0)


    # standardize the features
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # log parameters of the model:
    mlflow.log_params({"n_neighbors":5, "p":2})

    from sklearn.neighbors import KNeighborsClassifier
    # minkowski is for ecledian distance
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

    # Model training
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:,1]


    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    model_accuracy = accuracy_score(y_test,y_pred)

    mlflow.log_metric("accuracy",model_accuracy)
    print(f"model accuracy: {model_accuracy}")
