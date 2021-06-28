from sklearn import tree

#grande=1 #pequeno=2
#longo=1 #compacto=2

features = [[1, 1 ,2],
            [2, 2 ,1],
            [2, 2 ,1],
            [2, 1 ,0],
            [1, 1 ,0],
            [2, 1 ,1],
            [2, 1 ,0],
            [2, 2 ,1],
            [1, 1 ,2],
            [2, 1 ,1],
            [2, 1 ,2],
            [1, 1 ,0]]

#tesoura= 1 #porca=2 #parafuso=3 #caneta=4 #chave=5
labels = [1,2,2,3,4,5,3,2,1,5,5,4]

# o classificador encontra padrões nos dados de treinamento
clf = tree.DecisionTreeClassifier() # instância do classificador
clf = clf.fit(features, labels) # fit encontra padrões nos dados

# iremos utilizar para classificar uma nova fruta
print(clf.predict([[1, 1 ,1]]))

#grande1 #pequeno2
#longo1 #compacto2
#tesoura 1 #porca 2 #parafuso 3 #caneta 4 #chave5