import pandas as pd


def predict_placement(model, scaler, X_columns, input_data, X_reference_df):
    base = X_reference_df.mean().to_dict()
    base.update(input_data)

    input_df = pd.DataFrame([base])[X_columns]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    probability = (
        model.predict_proba(input_scaled)[0][1]
        if hasattr(model, "predict_proba")
        else None
    )

    return prediction, probability
