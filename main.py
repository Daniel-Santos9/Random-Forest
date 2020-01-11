#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 10:54:28 2019

@author: Daniel Santos
"""

from sklearn.ensemble import RandomForestClassifier 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Carregando os dados do data set.
base = load_breast_cancer()

#fazendoa separação de treinamento e teste
previsores_train, previsores_test, classe_train, classe_test = train_test_split(base.data, base.target, test_size=0.3)

#Fazendo a classificação
classificador = RandomForestClassifier(n_estimators=30 ,criterion='entropy' ,random_state=0)
classificador.fit(previsores_train,classe_train)

predict = classificador.predict(previsores_test)

acc_dataTrain = classificador.score(previsores_train, classe_train)

acc_dataTest = classificador.score(previsores_test,classe_test) 

acc = accuracy_score(classe_test, predict)

print(acc_dataTrain)
print(acc_dataTest)
print(acc)