import numpy as np


from sklearn.linear_model import SGDClassifier
from .models import CLFSwitcher


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


pipeline = Pipeline([
    ('clf', CLFSwitcher()),
])

randomness = np.arange(123, 124, 1)
parameters = [
	# {	
	# 	'clf__estimator__random_state': randomness,
	# 	'clf__estimator': [MLPClassifier()],
	# 	'clf__estimator__hidden_layer_sizes' : [100],
	# 	'clf__estimator__activation' : ['relu'],
	# 	'clf__estimator__tol': [1e-04, 1e-05, 1e-06, 1e-07, 1e-08,],
	# },
    {	
		'clf__estimator__random_state': randomness,
        'clf__estimator': [SGDClassifier()], # SVM if hinge loss / logreg if log loss
        'clf__estimator__penalty': ('l2', 'elasticnet', 'l1'),
        'clf__estimator__max_iter': [int(i) for i in [1e+03]],
		'clf__estimator__tol': [1e-05],
		# 'clf__estimator__alpha': [1e-00, 1e-02, 1e-04],
        'clf__estimator__loss': ['hinge', 'log_loss','perceptron', 'squared_hinge', 'modified_huber'],
    },
	# {
	# 	'clf__estimator__random_state': randomness,
	# 	'clf__estimator': [XGBRFClassifier(), XGBClassifier()],
	# 	'clf__estimator__max_depth': [5],
	# 	'clf__estimator__learning_rate': [1e-05], 
	# 	'clf__estimator__n_estimators': [int(i) for i in [1e+03]]
	# },
	# {
    #     'clf__estimator': [MultinomialNB()],
    #     'clf__estimator__alpha': (1e-2, 1e-3, 1e-1),
    # },
]

