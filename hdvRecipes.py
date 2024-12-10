import pandas as pd
import numpy as np
import csv
import torch
import torchhd

path = 'data/recipesData.csv'
vectorSize = 10000

def parseData(file):

    with open(path, 'r') as file:
        reader = csv.reader(file)
        max_columns = max(len(row) for row in reader)

    # Leer el archivo y rellenar filas con menos columnas usando NaN
    with open(path, 'r') as file:
        reader = csv.reader(file)
        rows = [row + [np.nan] * (max_columns - len(row)) for row in reader]

    # Convertir a DataFrame

    return pd.DataFrame(rows)
    
def main():
    
    '''Leeremos el fichero csv y lo transformaremos a un dataset
    que podamos usar y del cual obtendremos la información que necesitemos'''

    recipes = parseData(path)
    
    '''Una vez tenemos los datos del csv obtendremos los diferentes tipos de
    ingredientes que hay y los convertiremos en un hdv cada uno'''

    ingredients_data = recipes.iloc[:, 1:]  # Seleccionamos todas las columnas excepto la primera
    ingredients = ingredients_data.values.flatten() # Aplanamos los ingredientes en una lista unidimensional
    ingredients = ingredients[~pd.isna(ingredients)]  # Elimina NaN de la lista
    ingredients = pd.unique(ingredients) # Obtener los ingredientes sin repetir

    ingredientsHDV = {}
    for ingredient in ingredients:
        ingredientsHDV[ingredient] = torchhd.random(1, vectorSize, "MAP")

    '''Para codificar las recetas, simplemente serán el resultado de aplicar
    la operación de bundling con los diferentes ingredientes que incorpore la receta'''

    recipesHDV = []

    for recipe in recipes.iloc:
        recipeHDV = torchhd.bundle()
        recipesHDV.append(recipeHDV)

    # Número de recetas (número de filas)
    numRecipes = recipes.shape[0]

    #regions = recipes[0]
    #regions = regions.values.flatten()
    #regions = pd.unique(regions)
    
    print(recipes)
    print(recipes.iloc[1])
    #print(ingredients)
    #print(len(ingredients))
    pass

if __name__ == "__main__":
    main()