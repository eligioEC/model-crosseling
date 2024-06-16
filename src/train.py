# Importación de bibliotecas necesarias para la construcción de modelos
from sklearn import metrics
from sklearn.metrics import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.stats import randint
from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import pickle
import os
import pandas as pd

# Cargar data transformada y entrenar
def read_and_train(filename):
    df = pd.read_csv(os.path.join('../data/processed',filename))

    x_train = df.drop(['Compraron'],axis=1)
    y_train = df[['Compraron']]
    print(filename, " cargado correctamente")
    xgb_mod = XGBClassifier(max_depth=2,n_estimators=50,objective='binary:logistic',seed=0,silent=True,subsample=.8)
    xgb_mod.fit(x_train,y_train)
    print("modelo entrenado")
    # Guardamos el modelo entrenado en formato pkl
    package = '../models/best_model.pkl'
    pickle.dump(xgb_mod,open(package,'wb'))
    print("Modelo exportado correctamente en la carpeta models")

def main():
    read_and_train('cros_train.csv')
    print("Finalizo el entrenamiento del modelo")

if __name__ == "__main__":
    main()
    