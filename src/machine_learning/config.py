svr_param_grid = {
    'C': [1e-7, 1e-4, 1e-3, 1e-2, 1, 10, 100, 1000],
    'epsilon': [0.1, 0.2, 0.5, 0.3, 0.01, 0.001],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'sigmoid']
}

rf_param_grid = {
    'n_estimators': [100, 200, 400, 500, 750, 1000],
    'min_samples_split': [2, 3, 4, 5, 7, 10],
    'min_samples_leaf': [1, 2, 5]
}

rf_test_param_grid = {
    'n_estimators': [50],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

xtra_param_grid = {
    'n_estimators': [100, 200, 300, 400, 500, 750, 1000],
    'min_samples_split': [2, 3, 4, 5, 7, 10],
    'min_samples_leaf': [1, 2, 5]
}

gb_param_grid = {
    'n_estimators': [100, 200, 300, 500, 750, 1000],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 5],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
}

xgb_param_grid = {
    'n_estimators': [100, 200, 300, 500, 750, 1000],
    'min_samples_split': [3, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
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
