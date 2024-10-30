#arquivo de config

# Original Parameters
# LEARNING_RATE = 0.000003
# REGULARIZATION = 0.0002
# NUM_EPOCHS = 500
# BATCH_SIZE = 32
# IS_TRAINING = True

#configurações de ambiente
# !pip install numpy==1.23.5
# !pip install pandas==1.3.5
# !pip install tensorflow==2.8
# python 3.10

LEARNING_RATE = 0.0007
REGULARIZATION = 0.0002
NUM_EPOCHS = 200
BATCH_SIZE = 64
EARLY_STOPPING = 25
STEP_LR = LEARNING_RATE/100
IS_TRAINING = True

# variavel pra treinar ou carregar pesos, == True pra treinar e == False pra carregar pesos
TREINAR = False

path_input = "./data/input/battery-data-set" #onde esta os dados de entrada/treinamento
path_output = "./data/output/pesosteste.h5" #onde sera salvo os pesos e com qual nome
path_pesos = "./data/output/pesos.h5" #qual peso sera carregado

# testes
experiment0 = ["B0005", "B0006", "B0007", "B0018",
               "B0025", "B0026", "B0027", "B0028",
               "B0029", "B0030", "B0031", "B0032",
               "B0033", "B0034", "B0036", "B0038",
               "B0039", "B0040", "B0041", "B0042",
               "B0043", "B0044", "B0045", "B0046",
               "B0047", "B0048", "B0049", "B0050",
               "B0051", "B0052", "B0053", "B0054",
               "B0055", "B0056"]
experiment1 = ["B0005", "B0006", "B0007", "B0018"]
experiment2 = ["B0025", "B0026", "B0027", "B0028"]
experiment3 = ["B0029", "B0030", "B0031", "B0032"]
experiment4 = ["B0033", "B0034", "B0036"]
experiment5 = ["B0038", "B0039", "B0040"]
experiment6 = ["B0041", "B0042", "B0043", "B0044"]
experiment7 = ["B0045", "B0046", "B0047", "B0048"]
experiment8 = ["B0049", "B0050", "B0051", "B0052"]
experiment9 = ["B0053", "B0054", "B0055", "B0056"]