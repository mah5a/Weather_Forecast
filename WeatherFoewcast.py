import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
# print(df.head())

null_counts = df.isnull().sum()
print("---------------Nulls in each variable------------\n", null_counts)
# print(df.columns)
df = df.dropna()
print(df.columns)

# ---------Multivariate imputation by chained equations (MICE) method for null values-----------
# numeric_df = df.select_dtypes(include=[np.number])
# print(numeric_df.columns)
# imputer = IterativeImputer(random_state=0)
# imputed_data = imputer.fit_transform(numeric_df)
# imputed_df = pd.DataFrame(imputed_data, columns=numeric_df.columns)
# # imputed_df.info()
# imputed_df.to_csv(r"D:\mahsa\LEARNING\Weather_Forecast\imputed.csv")
# null_counts = imputed_df.isnull().sum()
# print(null_counts)
# --------------------------------------------------------------------------------------
