from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import BinaryCrossentropy, Huber
from tensorflow.keras.metrics import AUC, MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.initializers import he_normal
from config import MODEL_CONFIG, MODEL_SAVE_PATH, LOGS_DIR

def build_enhanced_model():
    """Улучшенная архитектура с раздельными ветками и регуляризацией"""
    input_layer = layers.Input(shape=(MODEL_CONFIG['input_size'],))
    
    # Основная ветвь с улучшенной регуляризацией
    x = layers.Dense(
        1024, 
        activation='swish',
        kernel_initializer=he_normal(),
        kernel_regularizer=regularizers.l2(0.005)
    )(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Независимые ветки для разных типов действий
    # Ветка для движений и стрельбы
    actions_branch = layers.Dense(
        512,
        activation='swish',
        kernel_regularizer=regularizers.l2(0.002)
    )(x)
    actions_branch = layers.Dense(256, activation='swish')(actions_branch)
    actions = layers.Dense(5, activation='sigmoid', name='actions')(actions_branch)
    
    # Ветка для прицеливания с тангенсной активацией
    aim_branch = layers.Dense(
        512,
        activation='tanh',
        kernel_initializer=he_normal()
    )(x)
    aim_branch = layers.Dense(256, activation='tanh')(aim_branch)
    aim = layers.Dense(2, activation='tanh', name='aim')(aim_branch)
    
    model = Model(inputs=input_layer, outputs=[actions, aim])
    
    # Оптимизатор с клиппингом градиентов
    optimizer = Nadam(
        learning_rate=MODEL_CONFIG['learning_rate'],
        clipvalue=0.5
    )
    
    # Компиляция с балансировкой потерь
    model.compile(
        optimizer=optimizer,
        loss={
            'actions': BinaryCrossentropy(),
            'aim': Huber(delta=0.2)
        },
        loss_weights={'actions': 0.5, 'aim': 0.5},
        metrics={
            'actions': AUC(name='pr_auc', curve='PR'),
            'aim': MeanAbsoluteError(name='mae')
        }
    )
    
    return model

def get_callbacks():
    """Улучшенная система колбэков"""
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=30,
            min_delta=0.005,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_aim_mae',
            factor=0.3,
            patience=10,
            cooldown=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            save_weights_only=False,
            save_best_only=True,
            monitor='val_aim_mae',
            mode='min'
        ),
        TensorBoard(
            log_dir=LOGS_DIR,
            histogram_freq=1,
            write_graph=True,
            update_freq=1000
        )
    ]