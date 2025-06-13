import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit


# load data and select training features
inspections = pd.read_csv("/Users/lakshaymaharana/Projects/honeybee-health/HoneybeeHealth/data/derived/Inspections_with_Weather.csv")
numerical_columns = inspections.select_dtypes(include=['number']).columns.tolist()
drop_columns = ['InspectionID', 'HiveID', 'Healthy_Binary', 'Percent_Met'] 
numerical_columns = [c for c in numerical_columns if c not in drop_columns]

drop_cat_columns = ['Date', 'Healthy', 'Hive_Tag', 'Apiary']
categorical_columns = inspections.select_dtypes(include=['object']).columns.tolist()
categorical_columns = [c for c in categorical_columns if c not in drop_cat_columns]


X = inspections[numerical_columns + categorical_columns]
y = inspections['Healthy_Binary']

# split data into train/test
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=inspections['HiveID']))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# create column transformer for preprocessing
preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

# pipeline to preprocess data and fit model aka determine coefficients of each feature
clf = Pipeline(steps=[
    ('prep', preprocess), 
    ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# fit the model on training data
clf.fit(X_train, y_train)

# test model against X_test and then we'll diff pred-test
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

