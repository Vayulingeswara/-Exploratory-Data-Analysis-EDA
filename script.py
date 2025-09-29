import pandas as pd
import numpy as np

# Load dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
gender_submission = pd.read_csv('gender_submission.csv')

# a. Data exploration mini guide
train_info = train.info()
train_desc = train.describe(include='all')

# Value counts on main categorical features
sex_value_counts = train['Sex'].value_counts()
pclass_value_counts = train['Pclass'].value_counts()
embarked_value_counts = train['Embarked'].value_counts()
survived_value_counts = train['Survived'].value_counts()

# Save all outputs as needed for further analysis
outputs = {
    'train_info': str(train_info),
    'train_describe': train_desc.to_dict(),
    'sex_value_counts': sex_value_counts.to_dict(),
    'pclass_value_counts': pclass_value_counts.to_dict(),
    'embarked_value_counts': embarked_value_counts.to_dict(),
    'survived_value_counts': survived_value_counts.to_dict()
}

outputs_str = str(outputs)
outputs_str[:4000] # Preview core outputs for summarization and further steps
