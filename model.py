import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
test_ids = test_df['Id']

train_df.drop("Id", axis=1, inplace=True)
test_df.drop("Id", axis=1, inplace=True)

X = train_df.drop("SalePrice", axis=1)
y = train_df["SalePrice"]

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))
])

param_dist = {
    'regressor__n_estimators': [800, 1200, 1600],
    'regressor__max_depth': [4, 6, 8, 10],
    'regressor__learning_rate': [0.01, 0.02, 0.05],
    'regressor__subsample': [0.7, 0.8, 1.0],
    'regressor__colsample_bytree': [0.7, 0.8, 1.0],
    'regressor__reg_alpha': [0, 0.1, 0.5],
    'regressor__reg_lambda': [0.5, 1, 2]
}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    model_pipeline,
    param_distributions=param_dist,
    n_iter=50,
    scoring='neg_root_mean_squared_error',
    cv=kfold,
    random_state=42,
    verbose=2,
    n_jobs=-1
)

search.fit(X, y)

best_model = search.best_estimator_

preds = best_model.predict(X)
rmse = mean_squared_error(y, preds, squared=False)
print(f"Training RMSE: {rmse:.2f}")

test_preds = best_model.predict(test_df)

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_preds
})
submission.to_csv("submission.csv", index=False)

joblib.dump(best_model, "house_price_model_xgboost.pkl")
print("Model saved.")
