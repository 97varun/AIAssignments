import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('selfie_dataset_header.txt', delim_whitespace=True)

pd.set_option('display.max_columns', 500)

features = ['is_female', 'baby', 'child', 'teenager', 'youth', 'middle_age', 'senior']
y_pos = np.arange(len(features))
counts = []

for feature in features:
    counts.append(df[feature].value_counts()[1])

plt.bar(y_pos, counts, align='center', alpha=0.5)
plt.xticks(y_pos, features)
plt.ylabel('counts')
plt.title('Data distribution before augmentation')

print(counts)

plt.savefig('dist_after.png')

# print(df.describe())