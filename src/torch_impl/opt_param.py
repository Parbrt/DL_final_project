from torch import dropout
from src.torch_impl.model_impl import CatModel, RegModel
import optuna

def reg_objective(trial):
    hidden_layer_1 = trial.suggest_int('hidden_layer_1',16,64)
    hidden_layer_2 = trial.suggest_int('hidden_layer_2',16,64)
    hidden_layer_3 = trial.suggest_int('hidden_layer_3',16,64)

    learning_rate = trial.suggest_float('lr',0.0001,0.1,log = True)
    drop_out = trial.suggest_float('dropout',0.0,0.75)
    pruning = trial.suggest_float('prunning',0.0,0.75)

    model = RegModel(hl1 = hidden_layer_1,
                     hl2 = hidden_layer_2,
                     hl3 = hidden_layer_3,
                     do = drop_out,
                     lr = learning_rate,
                     prune_amount = pruning)
    model.train()
    return model.loss.item()

def reg_optimization():
    study = optuna.create_study(
        study_name="optimization_csgo_torch",
        storage="sqlite:///../db.sqlite3",
        direction='minimize',
        load_if_exists=True
    )
    study.optimize(reg_objective,n_trials = 5)
    print("Best hyperparameters:",study.best_params)
    best_model = RegModel(hl1 = study.best_params['hidden_layer_1'],
                        hl2 = study.best_params['hidden_layer_2'],
                        hl3 = study.best_params['hidden_layer_3'],
                        do = study.best_params['dropout'],
                        lr = study.best_params['lr'],
                        prune_amount = study.best_params['prunning'])
    best_model.train()
    best_model.guess()
    best_model.get_metrics()

def cat_objective(trial):
    hidden_layer_1 = trial.suggest_int('hidden_layer_1',16,64)
    hidden_layer_2 = trial.suggest_int('hidden_layer_2',16,64)
    hidden_layer_3 = trial.suggest_int('hidden_layer_3',16,64)

    learning_rate = trial.suggest_float('lr',0.0001,0.1,log = True)
    drop_out = trial.suggest_float('dropout',0.0,0.75)
    pruning = trial.suggest_float('prunning',0.0,0.75)

    model = CatModel(hl1 = hidden_layer_1,
                     hl2 = hidden_layer_2,
                     hl3 = hidden_layer_3,
                     do = drop_out,
                     lr = learning_rate,
                     prune_amount = pruning)
    model.train()
    return model.loss.item()

def cat_optimization():
    study = optuna.create_study(
        study_name="optimization_winner_torch",
        storage="sqlite:///../db.sqlite3",
        direction='minimize',
        load_if_exists=True
    )
    study.optimize(cat_objective,n_trials = 5)
    print("Best hyperparameters:",study.best_params)
    best_model = CatModel(hl1 = study.best_params['hidden_layer_1'],
                        hl2 = study.best_params['hidden_layer_2'],
                        hl3 = study.best_params['hidden_layer_3'],
                        do = study.best_params['dropout'],
                        lr = study.best_params['lr'],
                        prune_amount = study.best_params['prunning'])
    best_model.train()
    best_model.guess()
    best_model.get_metrics()
