# script de preparacion de datos
################################

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# leemos la data
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/',filename)).set_index('id')
    df = df.rename(columns={"Response":"Compraron"})
    print(filename,' cargado correctamente')
    return df

def data_exporting(df, filename):
    df.to_csv(os.path.join("../data/processed/",filename),index=False)
    print(filename,"exprtado correctamente a la carpeta de prcessed")
    
# Realizando el encoding y el balanceo de data
def data_preparation(df):
    #Label Encoding
    le = LabelEncoder()

    df['Gender'] = le.fit_transform(df['Gender'])
    df['Vehicle_Damage'] = le.fit_transform(df['Vehicle_Damage'])
    

    #Ordinal Encoding
    df["Vehicle_Age"] = df["Vehicle_Age"].replace({"< 1 Year":0,"1-2 Year":1, "> 2 Years":2})

    if 'Compraron' in df.columns:
        #Balanceo de daos
        x = df.drop(['Compraron'], axis=1)
        y = df[['Compraron']]

        smote = SMOTE()
        x_smote, y_smote = smote.fit_resample(x,y)
        print("balanceo de datos realizado correcamente")
        print(y.value_counts())
        print()
        print(y_smote.value_counts())

        x_train, x_test, y_train,y_test = train_test_split(x_smote,y_smote,test_size=0.3, random_state=42)

        # Data train
        df_train = pd.concat([x_train, y_train], axis=1)
        print("data train")
        print(df_train.head())
        data_exporting(df_train,'cros_train.csv')

        # Data Val
        df_test = pd.concat([x_test, y_test], axis=1)
        data_exporting(df_test,'cros_test.csv')
        print("data test")
        print(df_train.head())
    else:
        print(df.head())
        data_exporting(df,'cros_score.csv')
        
    print("Tratamiento de data completado")

# Realizando el encoding y el balanceo de data
def data_preparation_and_scaling(df):
    #Label Encoding
    le = LabelEncoder()

    df['Gender'] = le.fit_transform(df['Gender'])
    df['Vehicle_Damage'] = le.fit_transform(df['Vehicle_Damage'])
    

    #Ordinal Encoding
    df["Vehicle_Age"] = df["Vehicle_Age"].replace({"< 1 Year":0,"1-2 Year":1, "> 2 Years":2})

    if 'Compraron' in df.columns:
        #Balanceo de daos
        x = df.drop(['Compraron'], axis=1)
        y = df[['Compraron']]

        smote = SMOTE()
        x_smote, y_smote = smote.fit_resample(x,y)
        print("balanceo de datos realizado correcamente")
        print(y.value_counts())
        print()
        print(y_smote.value_counts())

        #Fueature scaling
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_smote)

        x_train, x_test, y_train,y_test = train_test_split(x_scaled,y_smote,test_size=0.3, random_state=42)

        df_train = pd.DataFrame(x_train, columns=x.columns)
        df_test = pd.DataFrame(x_test, columns=x.columns)
        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        # Data train
        df_train = pd.concat([df_train, y_train], axis=1)
        print("data train")
        print(df_train.head())
        data_exporting(df_train,'cros_train.csv')

        # Data Val
        df_test = pd.concat([df_test, y_test], axis=1)
        data_exporting(df_test,'cros_test.csv')
        print("data test")
        print(df_train.head())
    else:
        #Fueature scaling
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(df)
        df_scaled = pd.DataFrame(x_scaled,columns=df.columns)
        df_scaled.reset_index(drop=True, inplace=True)
        print(df_scaled.head())
        data_exporting(df_scaled,'cros_score.csv')
        
    print("Tratamiento de data completado")

def main():
    # Matriz de entrenamiento y validacion
    df1 = read_file_csv('train.csv')
    #data_preparation(df1)
    data_preparation_and_scaling(df1)
    
    # Matriz de scoring
    df2 = read_file_csv('test.csv')
    #data_preparation(df2)
    data_preparation_and_scaling(df2)

if __name__ == "__main__":
    main()