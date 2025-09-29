import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load dataset
train = pd.read_csv('train.csv')

# b. sns.pairplot on selected numerical/categorical columns
pairplot = sns.pairplot(train[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']], hue='Survived', diag_kind='hist', palette='Set1',
                        plot_kws={'alpha': 0.6})
pairplot.savefig('pairplot.png')
plt.close()

# c. Heatmap for correlation
corr = train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr()
plt.figure(figsize=(8,6))
heatmap = sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('heatmap.png')
plt.close()

# d. Histograms, Boxplots, Scatterplots
# Histogram of Age
plt.figure(figsize=(6,4))
sns.histplot(train['Age'].dropna(), bins=30, kde=True)
plt.title('Distribution of Age')
plt.savefig('hist_age.png')
plt.close()

# Boxplot: Fare by Survived
plt.figure(figsize=(6,4))
sns.boxplot(x='Survived', y='Fare', data=train)
plt.title('Fare by Survival')
plt.savefig('box_fare_survived.png')
plt.close()

# Scatterplot: Age vs Fare colored by Survived
plt.figure(figsize=(6,4))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=train, palette='Set1')
plt.title('Age vs Fare by Survival')
plt.savefig('scatter_age_fare_survived.png')
plt.close()

# Save file output list
plot_files = {
    'pairplot': 'pairplot.png',
    'heatmap': 'heatmap.png',
    'hist_age': 'hist_age.png',
    'box_fare_survived': 'box_fare_survived.png',
    'scatter_age_fare_survived': 'scatter_age_fare_survived.png'
}

plot_files
