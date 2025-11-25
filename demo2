import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE


data = pd.read_csv("KDDTrain+.txt", sep=",", header=None)
print("Columns loaded:", list(data.columns))


label_col = data.columns[-2]       
difficulty_col = data.columns[-1]   


categorical_cols = [1, 2, 3]


X = data.drop(columns=[label_col, difficulty_col])
y_raw = data[label_col]


y = y_raw.apply(lambda v: 'normal' if v == 'normal' else 'attack')

print("Original label distribution:")
print(y_raw.value_counts())
print("\nBinary label distribution:")
print(y.value_counts())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


numeric_cols = [col for col in X_train.columns if col not in categorical_cols]


cat_train = X_train[categorical_cols].astype(str)

encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)
encoded_cat_train = encoder.fit_transform(cat_train)

cat_feature_names = encoder.get_feature_names_out(
    [str(c) for c in categorical_cols]
)

cat_train_df = pd.DataFrame(
    encoded_cat_train,
    columns=cat_feature_names,
    index=X_train.index
)


num_train = X_train[numeric_cols].astype(float)

scaler = StandardScaler()
scaled_num_train = scaler.fit_transform(num_train)

num_train_df = pd.DataFrame(
    scaled_num_train,
    columns=numeric_cols,
    index=X_train.index
)


X_train_final = pd.concat([num_train_df, cat_train_df], axis=1)
X_train_final.columns = X_train_final.columns.astype(str)

print("Train feature shape after preprocessing:", X_train_final.shape)


cat_test = X_test[categorical_cols].astype(str)
encoded_cat_test = encoder.transform(cat_test)

cat_test_df = pd.DataFrame(
    encoded_cat_test,
    columns=cat_feature_names,
    index=X_test.index
)

num_test = X_test[numeric_cols].astype(float)
scaled_num_test = scaler.transform(num_test)

num_test_df = pd.DataFrame(
    scaled_num_test,
    columns=numeric_cols,
    index=X_test.index
)

X_test_final = pd.concat([num_test_df, cat_test_df], axis=1)
X_test_final.columns = X_test_final.columns.astype(str)

print("Test feature shape after preprocessing:", X_test_final.shape)


smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_final, y_train)

print("Resampled train shape:", X_res.shape)
print("Class distribution after SMOTE:")
print(pd.Series(y_res).value_counts())


clf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

clf.fit(X_res, y_res)

joblib.dump(encoder, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(clf, 'random_forest_model.pkl')

joblib.dump({
    "categorical_cols": categorical_cols,
    "numeric_cols": numeric_cols,
    "cat_feature_names": cat_feature_names
}, "preprocessing_metadata.pkl")


y_pred = clf.predict(X_test_final)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
