import pandas as pd

def predict_pph(model, patient_data, all_features, important_features):

    data = {feature:0 for feature in all_features}

    for f in important_features:
        data[f] = patient_data[f]

    data_df = pd.DataFrame([data])

    probability = model.predict_proba(data_df)

    risk = probability[0][1] * 100

    if risk < 30:
        risk_level = "LOW RISK"
    elif risk < 70:
        risk_level = "MODERATE RISK"
    else:
        risk_level = "HIGH RISK"

    return risk, risk_level, data_df