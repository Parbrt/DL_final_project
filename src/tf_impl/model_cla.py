import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

def load_data():

    X_train = pd.read_csv('../data/processed/X_train.csv')
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_train = pd.read_csv('../data/processed/y_train_cat.csv')
    y_test = pd.read_csv('../data/processed/y_test_cat.csv')

    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, title="MODÈLE"):
    y_pred_prob = model.predict(X_test, verbose=0)

    results = model.evaluate(X_test, y_test, verbose=0)
    loss = results[0]
    accuracy = results[1]

    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"ACCURACY : {accuracy:.2%}")
    print(f"AUC ROC  : {auc:.4f}")
    print(f"LOSS     : {loss:.4f}")

def train_baseline(X_train, y_train):
    n_col = X_train.shape[1]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_col,)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=64, verbose=0)
    return model


def objective_optuna(trial, X_train, y_train):

    params = {
        'activation': trial.suggest_categorical('activation', ['relu', 'elu', 'swish']),
        'units_1': trial.suggest_int('units_1', 16, 100),
        'units_2': trial.suggest_int('units_2', 16, 100),
        'dropout_1': trial.suggest_float('dropout_rate_1', 0.0, 0.5),
        'dropout_2': trial.suggest_float('dropout_rate_2', 0.1, 0.5),
        'l2': trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True),
        'lr': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    }

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),

        tf.keras.layers.Dense(params['units_1'], activation=params['activation'],
                              kernel_regularizer=tf.keras.regularizers.l2(params['l2'])),
        tf.keras.layers.Dropout(params['dropout_1']),

        tf.keras.layers.Dense(params['units_2'], activation=params['activation'],
                              kernel_regularizer=tf.keras.regularizers.l2(params['l2'])),
        tf.keras.layers.Dropout(params['dropout_2']),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=15,
        batch_size=64,
        verbose=0
    )

    return history.history['val_loss'][-1]


def train_final_optimized(best, X_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),

        tf.keras.layers.Dense(best['units_1'], activation=best['activation'],
                              kernel_regularizer=tf.keras.regularizers.l2(best['l2_reg'])),
        tf.keras.layers.Dropout(best['dropout_rate_1']),

        tf.keras.layers.Dense(best['units_2'], activation=best['activation'],
                              kernel_regularizer=tf.keras.regularizers.l2(best['l2_reg'])),
        tf.keras.layers.Dropout(best['dropout_rate_2']),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    stopper = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=60,
        batch_size=32,
        callbacks=[stopper],
        verbose=1
    )

    return model
