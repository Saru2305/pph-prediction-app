import shap
import matplotlib.pyplot as plt
import numpy as np


def risk_explanation_panel(filtered_shap, important_features):

    print("\n---------------------------------")
    print("Patient Risk Explanation Panel")
    print("---------------------------------\n")

    impact = np.abs(filtered_shap)

    sorted_index = np.argsort(-impact)

    top_features = [important_features[i] for i in sorted_index[:5]]

    print("Top Factors Contributing to Risk:\n")

    for i,f in enumerate(top_features,1):
        print(f"{i}. {f}")


def shap_waterfall(filtered_shap, base_val, filtered_data, important_features):

    shap.plots.waterfall(
        shap.Explanation(
            values=filtered_shap,
            base_values=base_val,
            data=filtered_data,
            feature_names=important_features
        )
    )

    plt.show()


def feature_importance_chart(filtered_shap, important_features):

    plt.figure(figsize=(8,5))

    impact_values = np.abs(filtered_shap)

    sorted_idx = np.argsort(impact_values)

    plt.barh(
        [important_features[i] for i in sorted_idx],
        impact_values[sorted_idx]
    )

    plt.title("PPH Risk Feature Importance")
    plt.xlabel("Impact on Prediction")
    plt.ylabel("Features")

    plt.show()


def risk_gauge(risk):

    plt.figure(figsize=(6,6))

    labels = ['Low Risk','Moderate Risk','High Risk']
    sizes = [30,40,30]

    low_color='lightgrey'
    mod_color='lightgrey'
    high_color='lightgrey'

    if risk < 30:
        low_color='green'
    elif risk < 70:
        mod_color='orange'
    else:
        high_color='red'

    colors=[low_color,mod_color,high_color]

    plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        startangle=90,
        wedgeprops={'width':0.4}
    )

    plt.text(0,0,f"{risk:.1f}%",ha='center',va='center',fontsize=22,fontweight='bold')

    plt.title("PPH Risk Level Meter")

    plt.show()


