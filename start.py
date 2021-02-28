from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from classifier import knn, dt, nb

def cleandata(df):
    # Faz uma cópia do Dataframe original
    dfclean = df.copy(deep = True)
    # Mapeia 'tested_positive' => 1 e 'tested_negative' => 0
    dfclean['class'] = dfclean['class'].transform(lambda x: int(x == 'tested_positive'))
    # Remove Colunas com nulos
    nullables = [
        'plas',
        'pres',
        'skin',
        'insu',
        'mass'
    ]
    dfclean[nullables] = dfclean[nullables].replace(0, np.NaN)
    dfclean = dfclean.dropna()
    return dfclean

def test_knn(df):
    # Tabela de resultados do experimento
    table = {}
    x = df[['preg','plas','pres','skin','insu','mass','pedi','age']]
    y = df[['class']]
    for k in range(1,11):
        table[k] = {}
        for t in range(60,91,10):
            table[k][t] = []
            for i in range(1,21):
                print(f"Executing: k={k} treino={t}% iteração={i} ...")
                xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=((100-t)/100), random_state=None, stratify=y)
                xtrain = [ tuple(x) for x in xtrain.to_records(index=False) ]
                ytrain = list(ytrain['class'])
                xtest = [ tuple(x) for x in xtest.to_records(index=False) ]
                ytest = list(ytest['class'])
                results = knn(xtrain, ytrain, xtest, ytest, k = k)
                print(f"Acurácia: {results.accurracy()}")
                table[k][t].append(results.accurracy())
            table[k][t] = np.mean(table[k][t])
    return table
                
def test_nb(df):
    # Tabela de resultados do experimento
    table = {}
    x = df[['preg','plas','pres','skin','insu','mass','pedi','age']]
    y = df[['class']]
    for t in range(60,91,3):
        table[t] = []
        for i in range(1,41):
            print(f"Executing: treino={t}% iteração={i} ...")
            xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=((100-t)/100), random_state=None, stratify=y)
            results = nb(xtrain, ytrain, xtest, ytest)
            print(f"Acurácia: {results.accurracy()}")
            table[t].append(results.accurracy())
        table[t] = {
            'min': np.min(table[t]),
            'mean': np.mean(table[t]),
            'max': np.max(table[t])
        }
    return table

def test_dt(df):
    # Tabela de resultados do experimento
    table = {}
    x = df[['preg','plas','pres','skin','insu','mass','pedi','age']]
    y = df[['class']]
    for t in range(60,91,3):
        table[t] = []
        for i in range(1,41):
            print(f"Executing: treino={t}% iteração={i} ...")
            xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=((100-t)/100), random_state=None, stratify=y)
            results = dt(xtrain, ytrain, xtest, ytest)
            print(f"Acurácia: {results.accurracy()}")
            table[t].append(results.accurracy())
        table[t] = {
            'min': np.min(table[t]),
            'mean': np.mean(table[t]),
            'max': np.max(table[t])
        }
    return table

def make_accurracy_table(table, save=None, transpose=False):
    tabdf = pd.DataFrame(table)
    if transpose:
        tabdf = tabdf.T
    print("Tabela de Resultados")
    print(tabdf)
    if save:
        tabdf.to_csv(save)

def start(df, settings = {}):
    results = settings.get('results.path')
    print("===============")
    print("OBTENÇÃO")
    print("===============")
    print("Info:")
    df.info()
    print("---------------")
    print(df.describe().T)
    print("===============")
    print("LIMPEZA")
    print("===============")
    df = cleandata(df)
    df.to_csv('cleaned.csv')
    print(df.describe().T)
    print("===============")
    print("K-NN")
    print("===============")
    table = test_knn(df)
    make_accurracy_table(table, f"{results}/knn_results.csv")
    print("===============")
    print("NAIVE BAYES")
    print("===============")
    table = test_nb(df)
    make_accurracy_table(table, f"{results}/nb_results.csv", transpose = True)
    print("===============")
    print("DECISION TREE")
    print("===============")
    table = test_dt(df)
    make_accurracy_table(table, f"{results}/dt_results.csv", transpose = True)

