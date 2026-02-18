import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
car = pd.read_csv('Cleaned_Car_data.csv')

X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

# Preprocessing
ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

column_trans = ColumnTransformer([
    ('onehot', OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type'])
], remainder='passthrough')

# Model pipeline
model = Pipeline([
    ('preprocessor', column_trans),
    ('regressor', LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model.fit(X_train, y_train)

# Save NEW model
pickle.dump(model, open('LinearRegressionModel.pkl', 'wb'))

print("✅ Model retrained & saved successfully")
