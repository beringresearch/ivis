"""
Integrating ivis with standard sklearn pipelines
================================================

`Ivis` class extends sklearn's `BaseEstimator`, making it easy to incorporate ivis into a standard classification or regression pipeline.
"""

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from ivis import Ivis

# Make a toy dataset
X, y = make_classification(n_samples=1000,
        n_features=300, n_informative=250,
        n_redundant=0, n_repeated=0, n_classes=2,
        random_state=1234)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
        random_state = 1234)

ivis = Ivis(model = 'maaten', k = 10)
svc = LinearSVC(dual=False, random_state=1234)

clf_pipeline = Pipeline(steps=[('scaler', MinMaxScaler()),
                         ('ivis', ivis),
                         ('svc', svc)])
clf_pipeline.fit(X_train, y_train)

print("Accuracy on the test set with ivs transformation: {:.3f}".\
        format(clf_pipeline.score(X_test, y_test)))
