# Suchraum bewusst klein gehalten: rbf deckt die relevanten Faelle ab, sigmoid
# liefert hier praktisch nie das beste Modell. epsilon auf den Default reduziert,
# damit das jetzt serielle SVR den grossen Organismen nicht das Zeitbudget sprengt.
svr_param_grid = {
    'svr__C': [0.1, 1, 10, 100],
    'svr__epsilon': [0.1],
    'svr__kernel': ['rbf']
}

# n_estimators bei 700 bringt gegenueber 400 kaum Gewinn, kostet aber linear Zeit.
rf_param_grid = {
    'n_estimators': [200, 400],
    'min_samples_split': [2, 8],
    'min_samples_leaf': [1, 4]
}

rf_test_param_grid = {
    'n_estimators': [50],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

gb_param_grid = {
    'n_estimators': [300, 500, 800],
    'max_depth': [2, 3, 4],
    'subsample': [0.6, 0.8, 1.0],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2, 5],
    'learning_rate': [0.01, 0.05, 0.1],
}

# 16 Kombinationen. Beide Regularisierungsachsen (min_child_weight, subsample)
# bleiben, dafuer n_estimators fix auf 300, um das Zeitbudget zu halten.
xgb_param_grid = {
    'n_estimators': [300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 6],
    'min_child_weight': [1, 5],
    'subsample': [0.8, 1.0],
    # 'colsample_bytree': [0.6, 0.8, 1.0],
    # 'reg_lambda': [1.0, 3.0, 10.0],
    # 'reg_alpha': [0.0, 0.1, 1.0],
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

rf_cls_param_grid = {
    'n_estimators': [200, 400],
    'min_samples_split': [2, 8],
    'min_samples_leaf': [1, 4],
    'class_weight': [None, 'balanced'],
}

rf_cls_test_param_grid = {
    'n_estimators': [300],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
}

# 24 Kombinationen. class_weight bleibt wegen der Klassenungleichheit (WhiteLab
# ~16% positiv); max_depth-Cap und grosses min_samples_leaf als Regularisierung.
xtra_cls_param_grid = {
    'n_estimators': [300],
    'max_depth': [None, 20],
    'min_samples_split': [2, 8],
    'min_samples_leaf': [1, 4, 16],
    'max_features': ['sqrt'],
    'bootstrap': [False],   # typical for ExtraTrees
    'class_weight': [None, 'balanced']
}

gb_cls_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [300, 600, 1000],
    'max_depth': [2, 3, 4],
    'subsample': [0.6, 0.8, 1.0],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 1.0],
}

# 16 Kombinationen. Beide Regularisierungsachsen bleiben, n_estimators fix auf 300.
xgb_cls_param_grid = {
    'n_estimators': [300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 6],
    'min_child_weight': [1, 5],
    'subsample': [0.8, 1.0],
    # 'colsample_bytree': [0.6, 0.8, 1.0],
    # 'reg_lambda': [1.0, 3.0, 10.0],
    # 'reg_alpha': [0.0, 0.1, 1.0],
}

svc_cls_param_grid = {
    'svc__kernel': ['rbf'],
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': ['scale'],
    'svc__class_weight': [None, 'balanced'],
}


# 24 Kombinationen. max_depth-Cap und ein grosses min_samples_leaf geben dem Baum
# eine echte Bremse, gegen das starke Overfitting der Tree-Modelle (Train-R2 ~0,99).
xtra_param_grid = {
    'n_estimators': [300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 20],
    'min_samples_split': [2, 8],
    'min_samples_leaf': [1, 4, 16],
}


dummy_param_grid: dict = {}

ridge_param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0],
}

logreg_param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'class_weight': [None, 'balanced'],
}
