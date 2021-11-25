import numpy as np

input = np.ones((480,640,3))
print(input.shape)
input = np.transpose(input, (0,2,1))
print(input.shape)


"""
tic = time.time()
npList = np.array(list)
npListLabel = np.array(listLabel)
toc = time.time() - tic
print("Etapa 04 - Conversão Numpy Array - " + str(toc) + " segundos")

tic = time.time()
X_train, X_test, y_train, y_test = train_test_split(npList, npListLabel, test_size=0.2, random_state=42)
toc = time.time() - tic
print("Etapa 05 - Separação do dataset em treino e teste - " + str(toc) + " segundos")

tic = time.time()
pca = PCA(n_components=100)
npList = pca.fit_transform(npList)
npList = np.expand_dims(npList, axis=2)
toc = time.time() - tic
print("Etapa 06 - Redução de Dimensionalidade Utilizando PCA - " + str(toc) + " segundos")

tic = time.time()
plot_confusion_matrix(y_test, y_pred)
plt.savefig('word2vec2 - LSTM.png')
toc = time.time() - tic
print("Etapa 05 - Plotando Matrix de Confusão - " + str(toc) + " segundos")
"""