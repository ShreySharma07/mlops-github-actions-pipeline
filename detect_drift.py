import pandas as pd
from sklearn.metrics import jaccard_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

raw_data = pd.read_csv('data/raw_data.csv')
production_data = pd.read_csv('data/production_data.csv')

print(production_data.columns)

features = ['0', '1']

for feature in features:
    print(f'\n drift analysis for features')

    print(f'training data {feature} stats : {raw_data[feature].describe}')
    print(f'production data {feature} stats : {production_data[feature].describe}')

    plt.figure(figsize=(10, 5))
    sns.kdeplot(raw_data[feature], label = 'raw_data', color = 'blue')
    sns.kdeplot(production_data[features], label = 'production_data', color = 'red')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

    statistic, p_value = ks_2samp(raw_data[feature], production_data[feature])
    print(f"KS Test → statistic: {statistic:.4f}, p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("⚠️  Drift Detected!")
    else:
        print("✅ No significant drift detected.")