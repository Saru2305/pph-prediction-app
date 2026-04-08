import shap

def shap_explanation(model, data_df):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(data_df)

    if isinstance(shap_values, list):

        shap_val = shap_values[1][0]
        base_val = explainer.expected_value[1]

    else:

        shap_val = shap_values[0]
        base_val = explainer.expected_value

    return shap_val, base_val