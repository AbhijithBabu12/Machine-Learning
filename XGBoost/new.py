import pandas as pd
df = pd.read_csv('titanic.csv')

X = df.drop('Survived', axis = 1)
y = df['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print('Accuracy :', accuracy_score(y_test, y_pred))