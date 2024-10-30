#função de simplificação, serve para treinar o modelo
from sklearn.model_selection import train_test_split
import tensorflow as tf
from config import  BATCH_SIZE, EARLY_STOPPING, IS_TRAINING, NUM_EPOCHS, STEP_LR
from config import path_output as diretorio
from modeloRUL import get_model
from processar_dados import get_exp_based_df
from separar_treino_teste import separar



def learn(experimento) :
    # busca os dados dos arquivos .mat usando a funcao get_exp_based_df, df_x = caracteristicas, df_y = capacidade
    train_x, train_y, test_x, test_y, val_x, val_y = separar(experimento)

    #definindo a estrutura de dados que o modelo vai receber
    input_shape = (train_x.shape[1], train_x.shape[2])

    #construindo o modelo
    model = get_model(input_shape, is_bidirectional=False)

    if IS_TRAINING:
        def scheduler(epoch, lr):
            if epoch < 10:
                return lr+STEP_LR
            elif epoch % 5 == 0:
                return lr*0.99
            return lr

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=EARLY_STOPPING)
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

        model.fit(train_x, train_y, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                            verbose=1, validation_split=0.1, callbacks=[early_stopping, lr_scheduler])
    
    #avalia o modelo
    model.evaluate(train_x, train_y)
    model.evaluate(val_x, val_y)
    model.evaluate(test_x, test_y)
        
    return model