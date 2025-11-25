import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset without headers
data = pd.read_csv("KDDTrain+.txt", sep=",", header=None)
print("Columns loaded:", list(data.columns))

# Specify categorical columns by column indices, adjust these indices based on your dataset
categorical_cols = [1, 2, 3, data.columns[-2]]  # e.g., protocol, flag, status, category columns
label_col = data.columns[-1]  # usually the last column is label

# Convert column indices to string for encoder's get_feature_names_out compatibility
cat_col_str = [str(col) for col in categorical_cols]

# Create and fit one-hot encoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cat = encoder.fit_transform(data[categorical_cols])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_col_str))

# Drop original categorical columns and concatenate encoded columns
data = data.drop(columns=categorical_cols).reset_index(drop=True)
data = pd.concat([data, encoded_cat_df], axis=1)

# Ensure all columns names are string type for scaler compatibility
data.columns = data.columns.astype(str)
label_col = str(label_col)

# Scale all numeric columns except the target/label
numeric_cols = [col for col in data.select_dtypes(include=np.number).columns if col != label_col]
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Separate features and labels
features = data.drop(columns=[label_col])
labels = data[label_col]

# Split into train and test datasets, stratify by label to maintain class proportion
X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    test_size=0.2,
                                                    stratify=labels,
                                                    random_state=42)

# Apply SMOTE to oversample minority classes on training data
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Train RandomForestClassifier with class_weight balanced and parallel jobs
clf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
clf.fit(X_res, y_res)

# Save encoder, scaler, and model for later use
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(clf, 'random_forest_model.pkl')

# Predict and evaluate on test data
y_pred = clf.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
