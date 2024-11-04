config_ch = Config(
	RANDOM_STATE=42,
	TASK = 'classification',
	DATASET='churn',X_COL = 'market',
	CRITERION='gini', 
	MAX_DEPTH=3,
	BASE_SCORE=0.5, 
	BOOSTER='gbtree',
	N_ESTIMATORS=1000,
	OBJECTIVE='binary:logistic',
	LR=1e-04
)

def conf_eg(y_col='conversion'):
    config_eg = Config(
        RANDOM_STATE=42,
        TASK = 'classification',
        DATASET='engagement', X_COL = 'market', Y_COL = y_col,
        CRITERION='gini', 
        MAX_DEPTH=3,
        BASE_SCORE=0.5, 
        BOOSTER='gbtree',
        N_ESTIMATORS=1000,
        OBJECTIVE='binary:logistic',
        LR=1e-04
    )
    return config_eg