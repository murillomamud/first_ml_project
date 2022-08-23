import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


from keras.models import Sequential
from keras.layers import Dense


dataset = pd.read_csv('Churn_Modelling.csv')

#X = VARIÁVEIS INDEPENDENTES
X = dataset.iloc[:,3:13].values

#Y = VARIÁVEIS QUE EU QUERO PREVER (DEPENDENTES)
y = dataset.iloc[:,13].values


#transformar dados categóricos em dados numéricos
#país em números
labelEncoder_X_1 = LabelEncoder()
X[:,1] = labelEncoder_X_1.fit_transform(X[:,1])

#genero em números
labelEncoder_X_2 = LabelEncoder()
X[:,2] = labelEncoder_X_2.fit_transform(X[:,2])

#colocar a categoria país em colunas

oneHotEncoder = make_column_transformer((OneHotEncoder(categories='auto', sparse=False),[1]), remainder="passthrough")

X = oneHotEncoder.fit_transform(X)

#remove 1 coluna de país para evitar Dummy Variable Trap.
X = X[:,1:]


#DIVIDIR DADOS ENTRE TREINO E TESTE
X_training, X_test, y_training, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#print(len(X_training), len(X_test), len(y_training), len(y_test))

#PADRONIZAR OS DADOS PARA CRIAÇÃO DO MODELO
sc = StandardScaler()
X_training = sc.fit_transform(X_training)
X_test = sc.fit_transform(X_test)

#inicializar rede neural
classifier = Sequential()

#criação de camadas, units = numero de neuronios, inicialização método uniform, função de ativação relu, input_dim = camada de entrada, ou seja 11 nós na camada de entrada(numero de atributos dos dados)
#como definir numero de neuronios camada oculta - média dos neuronios da camada de entrada com a camada de saída
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

#criar mais uma camada oculta
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

#criar camada de saída - 1 neurônio, pois, minha saída vai dizer apenas 0 ou 1
#como quero probablidade, utilizar funcção de ativação sigmoid
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

#compilação do modelo
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#treinar a rede
#batch_size = numero registros serão passados pela rede antes de atualizar pesos
#epochs = quantas vezes os dados vao passar pela rede antes de encerrar treinamento
classifier.fit(X_training, y_training, batch_size=10, epochs=100)


#previsao dos dados de teste
y_pred = classifier.predict(X_test)

#passar de % para 0 ou 1
y_pred = (y_pred > 0.5)

#criação de matriz de confusão (previsto x realizado)
cm = confusion_matrix(y_test, y_pred)

print(cm)

#para verificar taxa de acerto.. acertos / total dados teste




