import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train_reg = pd.read_csv('data/processed/y_train_reg_scaled.csv')
    y_test_reg = pd.read_csv('data/processed/y_test_reg_scaled.csv')

    df_raw = pd.read_csv('data/csgo_round_snapshots.csv')
    y_reg_raw = df_raw[['ct_money', 'ct_health']]

    _, _, y_train_real, y_test_real = train_test_split(
        df_raw, y_reg_raw, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    scaler.fit(y_train_real)

    return X_train, X_test, y_train_reg, y_test_reg, scaler, y_test_real


def evaluate_model(model, X_test, scaler, y_test_real, title="REG"):
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred_real = scaler.inverse_transform(y_pred_scaled)

    r2 = r2_score(y_test_real, y_pred_real, multioutput='raw_values')
    mae = mean_absolute_error(y_test_real, y_pred_real, multioutput='raw_values')

    print(f"ARGENT : Erreur {mae[0]:.0f} dollars  (R²: {r2[0]:.2f})")
    print(f"PV     : Erreur {mae[1]:.0f} HP (R²: {r2[1]:.2f})")


def train_baseline(X_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=64, verbose=0)
    return model


def objective_optuna(trial, X_train, y_train):
    params = {
        'units_1': trial.suggest_int('units_1', 16, 128),
        'units_2': trial.suggest_int('units_2', 16, 128),
        'activation': trial.suggest_categorical('activation', ['relu', 'elu', 'swish']),
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
        tf.keras.layers.Dense(2, activation='linear')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']),
                  loss='mse', metrics=['mae'])

    history = model.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=64, verbose=0)
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
        tf.keras.layers.Dense(2, activation='linear')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best['learning_rate']),
                  loss='mse', metrics=['mae'])

    stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=64,
              callbacks=[stopper], verbose=1)
    return model
