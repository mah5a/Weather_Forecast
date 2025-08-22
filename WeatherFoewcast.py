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


df = df.rename(columns={'RainToday': 'RainYesterday', 'RainTomorrow': 'RainToday'})
df = df[df.Location.isin(['Melbourne', 'MelbourneAirport', 'Watsonia', ])]
print(df.info())


# Create a function to map dates to seasons
def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'


df['Date'] = pd.to_datetime(df['Date'])
df['Season'] = df['Date'].apply(date_to_season)
df = df.drop(columns='Date')
print(df)
X = df.drop(columns='RainToday', axis=1)
y = df['RainToday']

print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Define separate transformers for both feature types
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
# Scale the numeric features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# One-hot encode the categoricals
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine the transformers into a single preprocessing column transformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
]
)
# Create a pipeline by combining the preprocessing with a Random Forest classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define a parameter grid to use in a cross validation grid search model optimizer
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}
# Perform grid search cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    verbose=2
)

grid_search.fit(X_train, y_train)
test_score = grid_search.score(X_test, y_test)
print("Test set score: {:.2f}".format(test_score))