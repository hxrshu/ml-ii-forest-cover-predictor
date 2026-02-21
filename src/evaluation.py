from sklearn.model_selection import StratifiedKFold, cross_validate

def evaluate_model_performance(model, X, y, n_folds=5, n_jobs=-1):
    kf = StratifiedKFold(n_splits=n_folds)
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring="balanced_accuracy",
        return_train_score=True,
        n_jobs=n_jobs,
    )
    train_score = cv_results["train_score"].mean()
    validate_score = cv_results["test_score"].mean()
    print(f"Train Balanced Accuracy: {train_score:.4f}")
    print(f"Validate Balanced Accuracy: {validate_score:.4f}")
    return train_score, validate_score