import joblib
import pandas as pd

pipeline = joblib.load("models/pipeline_full.pkl")

# 🔥 Load training columns properly
expected_cols = list(pipeline.feature_names_in_)

def predict(data):
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    elif not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be DataFrame or dict")

    # 🔥 Create empty dataframe with ALL columns
    full_data = pd.DataFrame(columns=expected_cols)

    # 🔥 Fill provided values
    for col in data.columns:
        full_data[col] = data[col]

    # 🔥 Keep correct order
    full_data = full_data[expected_cols]

    pred = pipeline.predict_proba(full_data)[:, 1]
    return pred.tolist()


if __name__ == "__main__":
    sample = pd.read_csv("application_train.csv").iloc[:1]
    print("Prediction:", predict(sample)[0])
