import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

import ml.data as data
import ml.model as model
from config.logging_config import info_log

info_log("Training pipeline: Started.")

df = pd.read_csv("/app/data/cleaned_data.csv")

info_log("Training: Data split and encoding.")
train, test = train_test_split(df, test_size=0.20)

X_train, y_train, encoder, lb = data.process_data(
    train, categorical_features=model.CAT_FEATURES, label="salary", training=True
)

info_log("Training: model training.")

trained_model = model.train_model(X_train, y_train)

info_log("Training: saving model and encoder to disk.")

with open('artifacts/regressor_model.pkl', 'wb') as f:
    pickle.dump(trained_model, f)

with open('artifacts/encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

info_log("Training: generating inference values.")
y_preds = model.inference(trained_model, X_train)

model.compute_model_metrics(y_train, y_preds)

info_log("Training: computing sliced data metrics.")
metrics = {}

for category in model.CAT_FEATURES:
    metrics[category] = {}
    unique_values = train[category].unique()

    for value in unique_values:
        distinct_rows = train[category] == value
        y_pred_slice = y_preds[distinct_rows]
        y_actual_slice = y_train[distinct_rows]

        precision, recall, fbeta = model.compute_model_metrics(
            y_actual_slice, y_pred_slice)

        metrics[category][value] = {'precision': precision,
                                    'recall': recall, 'fbeta': fbeta}


with open("sliced_data_metrics", "w") as file:
    for category in metrics:
        file.write(f"Metrics for slicing by {category}: \n\n")

        for values in metrics[category]:
            file.write(f"{values}: {metrics[category][values]} \n")

        file.write("------------ \n")
