from config import  TREINAR, experiment0
from config import path_output as diretorio
from ensinar_modelo import learn
from iniciar_modelo_pesos import carregar_pesos

#arquivo de execução

if __name__ == "__main__":
    if(TREINAR) :
        model = learn(experiment0)
        #salva os pesos do modelo no diretorio
        model.save_weights(diretorio)
        print(f"salvo no diretorio: {diretorio}")
    else :
        carregar_pesos(experiment0)