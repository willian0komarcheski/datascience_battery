# separa os dados retornados pelas função get_exp_based_df para dados de treino e dados de teste e depois converte os dados para float32
from sklearn.model_selection import train_test_split
from processar_dados import get_exp_based_df


def separar(experimento) :
    df_x, df_y = get_exp_based_df(experimento)

    #separando os dados de treino e de teste
    train_x, test_x, train_y, test_y = train_test_split(
        df_x, df_y, test_size=0.2, random_state=0)

    #separando os dados de teste mais uma vez
    test_x, val_x, test_y, val_y = train_test_split(
        test_x, test_y, test_size=0.5, random_state=0)

    #convertendo os dados para float32
    train_x = train_x.astype('float32')
    train_y = train_y.astype('float32')
    val_x = val_x.astype('float32')
    val_y= val_y.astype('float32')
    test_x = test_x.astype('float32')
    test_y = test_y.astype('float32')

    return train_x, train_y, test_x, test_y, val_x, val_y