svr_param_grid = {
    'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3],
    'epsilon': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'sigmoid']
}

rf_param_grid = {
    'n_estimators': [200, 400, 750, 1000],
    'min_samples_split': [2, 4, 8],
    'min_samples_leaf': [1, 2, 5],
    'max_depth': [None, 8, 12, 20],
}

rf_test_param_grid = {
    'n_estimators': [50],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

xtra_param_grid = {
    'n_estimators': [300, 500, 750, 1000],
    'max_depth': [None, 12, 20],
    'min_samples_split': [3, 4, 5, 7],
    'min_samples_leaf': [1, 2, 5]
}

gb_param_grid = {
    'n_estimators': [300, 500, 800],
    'max_depth': [2, 3, 4],
    'subsample': [0.6, 0.8, 1.0],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2, 5],
    'learning_rate': [0.01, 0.05, 0.1],
}

xgb_param_grid = {
    'n_estimators': [400, 800, 1200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_lambda': [1.0, 3.0, 10.0],
    'reg_alpha': [0.0, 0.1, 1.0],
}

TOP_ORGANISMS = [
    "Staphylococcus aureus",
    "Escherichia coli",
    "Pseudomonas aeruginosa",
    "Candida albicans",
    "Bacillus subtilis",
    "Klebsiella pneumoniae",
    "Staphylococcus epidermidis",
    "Acinetobacter baumannii",
    "Enterococcus faecalis",
    "Micrococcus luteus",
    "Salmonella enterica",
    "Enterobacter cloacae",
    "Enterobacter aerogenes",
    "Enterobacter sp."
]
