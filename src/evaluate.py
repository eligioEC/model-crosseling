# Codigo de evaluacion - model crosselling
##########################################

import pandas as pd
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import *
import os

# Carta la data de evaluacion
def eval_model(filename):
    df = pd.read_csv(os.path.join("../data/processed",filename))
    print(filename, " cargado correctamente")
    # Leemos el modelo entranado para usarlo
    package = "../models/best_model.pkl"
    model = pickle.load(open(package,'rb'))
    print("modelo importado correctamente")
    # Predecimos sobre el set de datoa de validacion
    x_test = df.drop(['Compraron'],axis=1)
    y_test = df[['Compraron']]
    y_pred_test = model.predict(x_test)
    # Generamos metricas de diagnostico
    cm_test = confusion_matrix(y_test,y_pred_test)
    print("Matriz de confusion: ")
    print(cm_test)
    accuracy_test = accuracy_score(y_test,y_pred_test)
    print("Accuracy: ",accuracy_test)
    precision_test = precision_score(y_test,y_pred_test)
    print("Precision: ",precision_test)
    recall_test = recall_score(y_test,y_pred_test)
    print("Recall: ",recall_test)

def main():
    eval_model('cros_test.csv')
    print("Finalizo la validacion del modelo")

if __name__ == "__main__":
    main()
    