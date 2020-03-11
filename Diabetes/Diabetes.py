# %% [markdown]
# # Predicting Diabetes Readmission
# 
# Author: ...
# 
# ## Introduction
# 
# The goal of this data analysis report is to predict whether and how soon a diabetes patient would be readmitted to the hospital based on demographics as well as visit and medication details.

# %%
import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

# %% [markdown]
# ## Data
# 
# The data resides in the file `diabetes.tab` (MD5 after cleaning: `80521F2659B0CF928EB7C2F325B8D95B`) in the `dat` subfolder and was originally collected from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008) and then adjusted as follows:
# 
# 1. Select and rename features (see `collect.r` for details)
# 2. Forcing the features `Source` and `Diagnosis` to be read as categorical instead of numerical variables
# 3. Converting values `?`, `Unknown/Invalid`, … to empty values (see `collect.r` for details)
# 4. Convert separator comma's to tabs
# %% [markdown]
# ## Loading the Data
# 
# The retained data set is split into 2 parts, one for training (&frac23;) and one for testing (&frac13;). The seed was set to 42 to allow for reproduction of this work.

# %%
data_df = pd.read_csv("dat/diabetes.tab", sep="\t")

# %% [markdown]
# ## Creating Dummy Variables
# 
# Although Decision trees inheritly support mixed data types, Python SKLearn does not. To work around this, we will convert the categorical features in our data set into corresponding dummy variables: 

# %%
data_df = pd.get_dummies (data_df, columns=["Race", "Gender", "Age", "Source", "Diagnosis",
        "Diagnosis_Count", "HbA1C", "Metformin", "Repaglinide", "Nateglinide", "Chlorpropamide",
        "Glimepiride", "Acetohexamide", "Glipizide", "Glyburide", "Tolbutamide", "Pioglitazone",
        "Rosiglitazone", "Acarbose", "Miglitol", "Troglitazone", "Tolazamide", "Examide",
        "Citoglipton", "Insulin", "Glyburide_metformin", "Glipizide_metformin",
        "Glimepiride_pioglitazone", "Metformin_rosiglitazone", "Metformin_pioglitazone"])

feat = data_df[set(data_df.columns.values).difference({"Readmission"})]
outc = data_df.Readmission

# %% [markdown]
# ## Splitting the Data
# 
# The data will be split into a training set (1/3), a validation set (1/3) and a testing set (1/3)

# %%
feat_trai_vali, feat_test, outc_trai_vali, outc_test = train_test_split(
    feat, outc, test_size=0.33, random_state=42)
feat_trai, feat_vali, outc_trai, outc_vali = train_test_split(
    feat_trai_vali, outc_trai_vali, test_size=0.50, random_state=42)

# %% [markdown]
# ## Feature Contribution
# %% [markdown]
# ## Hyperparameters
# 
# 
# %% [markdown]
# ## Best Model
# %% [markdown]
# ## Prediction

# %%
outc_pred = learner.predict(feat_test)
outc_pred_prob = learner.predict_proba(feat_test)[:, 1]

list(zip(outc_pred, outc_test))

# %% [markdown]
# ## Performance
# 
# The ROC curve below demonstrates the classification success independent of a threshold.

# %%
f"Performance of the decision tree classifier {learner.score(feat_test, outc_test):.2%} (by chance = 33.33%)"

# %% [markdown]
# ## Conclusion
