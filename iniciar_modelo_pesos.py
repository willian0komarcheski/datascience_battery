#cria o modelo para depois carregar os pesos nele
from sklearn.model_selection import train_test_split
from modeloRUL import get_model
from processar_dados import get_exp_based_df
from separar_treino_teste import separar
from config import path_pesos as diretorio

def carregar_pesos(experimento) :

    train_x, train_y, test_x, test_y, val_x, val_y = separar(experimento)

    input_shape = (train_x.shape[1], train_x.shape[2])

    new_model = get_model(input_shape, is_bidirectional=False)

    #carrega os pesos do modelo salvado no diretorio
    new_model.load_weights(diretorio)
    
    new_model.evaluate(train_x, train_y)
    new_model.evaluate(val_x, val_y)
    new_model.evaluate(test_x, test_y)

    return new_model