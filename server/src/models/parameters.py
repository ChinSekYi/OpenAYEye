from xgboost import XGBClassifier, XGBRFClassifier

randomness = (123,)
parameters = [
	{
		'clf__estimator__random_state': randomness,
		'clf__estimator': [XGBClassifier(tree_method='hist', enable_categorical=True)],
		'clf__estimator__max_depth': [5],
		'clf__estimator__learning_rate': [1e-05], 
		'clf__estimator__n_estimators': [int(i) for i in [1e+03]],
	},
]