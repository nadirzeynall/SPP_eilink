import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

mlflow.autolog()

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)

import os
from random import random, randint
from mlflow import log_metric, log_param, log_params, log_artifacts

if __name__ == "__main__":
    # Log a parameter (key-value pair)
    log_param("config_value", randint(0, 100))

    # Log a dictionary of parameters
    log_params({"param1": randint(0, 100), "param2": randint(0, 100)})

    # Log a metric; metrics can be updated throughout the run
    log_metric("accuracy", random() / 2.0)
    log_metric("accuracy", random() + 0.1)
    log_metric("accuracy", random() + 0.2)

    # Log an artifact (output file)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")
    log_artifacts("outputs")

    mlflow.set_tracking_uri("http://10.201.10.118:5000")