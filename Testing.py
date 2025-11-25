
import pandas as pd
import joblib
from sklearn import accuracy_score, classification_report, confusion_matrix
clf = joblib.load(random_forest_model.pkl)
encoder= joblib.load('encoder.pkl')
scaler= joblib.load('scaler.pkl')

test_data = pd.read_csv("KDDTest+", sep=",")
test_data= test_data.drop(['FlowID','Timestamp'],axis=1,errors='ignore')

protocol_encoded_test = encoder.transform(test_data[['tcp','ftp_data','SF','normal']])
protocol_df_test=pd.DataFrame(protocol_encoded_test, columns=encoder.get_feature_names_out(['tcp','ftp_data','SF','normal']))

test_data.reset_index(drop=True, inplace=True)
test_data=pd.concat([test_data,protocol_df_test],axis=1)
test_data=test_data.drop(['tcp','ftp_data','SF','normal'],axis=1)

numeric_cols_test= test_data.select_dtypes(include=numbers).coloum