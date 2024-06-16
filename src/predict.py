# Codigo de scoring - modelo crosselling
#=========================================
import pandas as pd
import xgboost as xgb
import pickle
import os

# Cargar data para predecir
def score_model(filename,scores):
    df = pd.read_csv(os.path.join("../data/processed",filename))
    print(filename, " cargado correctamente")
    print(df.info())
    # Leemos el modelo entrenado
    package = "../models/best_model.pkl"
    model = pickle.load(open(package,'rb'))
    print("Modelo importado correctamente")
    # Predecimos sobre el set de datos de scoring
    res = model.predict(df).reshape(-1,1)
    df_pred = pd.DataFrame(res,columns=["predict"])
    df_pred.to_csv(os.path.join('../data/scores/', scores))
    print(scores, ' exportado correctamente en la carpeta scores')

def main():
    score_model('cros_score.csv','final_score.csv')
    print("Finalizo el scoring del modelo")

if __name__ == "__main__":
    main()
    
    