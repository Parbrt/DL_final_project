import optuna

import tf_impl.model_reg as reg_module
import tf_impl.model_cla as clf_module


def run_regression_tf():
    X_train, X_test, y_train, y_test, scaler, y_real_test = reg_module.load_data()

    model_base = reg_module.train_baseline(X_train, y_train)
    reg_module.evaluate_model(model_base, X_test, scaler, y_real_test, title="BASELINE REG")

    study = optuna.create_study(
        study_name="optimization_csgo_reg",
        storage="sqlite:///../db.sqlite3",
        direction='minimize',
        load_if_exists=True
    )
    study.optimize(lambda trial: reg_module.objective_optuna(trial, X_train, y_train), n_trials=20)

    model_final = reg_module.train_final_optimized(study.best_params, X_train, y_train)
    reg_module.evaluate_model(model_final, X_test, scaler, y_real_test, title="FINAL REG OPTIMISÉ")


def run_classification_tf():
    X_train, X_test, y_train, y_test = clf_module.load_data()

    model_base = clf_module.train_baseline(X_train, y_train)
    clf_module.evaluate_model(model_base, X_test, y_test, title="BASELINE CLF")

    study = optuna.create_study(
        study_name="optimization_csgo_clf",
        storage="sqlite:///../db.sqlite3",
        direction='minimize',
        load_if_exists=True
    )
    study.optimize(lambda trial: clf_module.objective_optuna(trial, X_train, y_train), n_trials=20)

    model_final = clf_module.train_final_optimized(study.best_params, X_train, y_train)
    clf_module.evaluate_model(model_final, X_test, y_test, title="FINAL CLF OPTIMISÉ")

