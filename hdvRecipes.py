import pandas as pd
import numpy as np
import csv
import torchhd

path = '/home/victor/hdv/hdvExamples/data/recipesData.csv'
#path = 'data/recipesData.csv'
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

def encodeRecipe(ingredients, ingredientsHDV):
    newRecipeHDV = []
    for i in range(len(ingredients)):
        if i == 0:
            newRecipeHDV = ingredientsHDV.get(ingredients[i])
        else:
            newRecipeHDV = torchhd.bundle(newRecipeHDV,ingredientsHDV.get(ingredients[i]))

    return newRecipeHDV.normalize()

def decodeRecipe(recipeHDV, recipesSampleHDV, recipeSample):
    bestSimilarRecipe = ""
    bestSimilarity = -1

    for i in range(len(recipesSampleHDV)):
        similarity = torchhd.cosine_similarity(recipeHDV, recipesSampleHDV[i])
        if similarity > bestSimilarity:
            bestSimilarity = similarity
            bestSimilarRecipe = recipeSample.iloc[i].dropna()

    print(bestSimilarity)
    return bestSimilarRecipe

def encodeRegion(region, recipesSampleHDV, recipeSample):
    regionHDV = []
    for i in range(recipeSample.shape[0]): # Con esto recorreremos todas las recetas
        recipeRegion = recipeSample.iloc[i,0] # Obtenemos la región de la receta
        if recipeRegion == region and regionHDV == []:
            regionHDV = recipesSampleHDV[i]
        elif recipeRegion == region:
            regionHDV = torchhd.bundle(regionHDV, recipesSampleHDV[i]) # Según hemos recorrido las recetas las hemos guardado, por eso sabemos el orden
    
    return regionHDV.normalize()

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

    ingredientsHDV = {} # Diccionario para almacenar el ingrediente y su HDV
    for ingredient in ingredients:
        ingredientsHDV[ingredient] = torchhd.random(1, vectorSize, "MAP")

    '''Para codificar las recetas, simplemente serán el resultado de aplicar
    la operación de bundling con los diferentes ingredientes que incorpore la receta'''

    recipesHDV = []

    for i in range(recipes.shape[0]): # Con esto recorreremos todas las filas, es decir, todas las recetas encodeRecipe(ingredients, ingredientsHDV)
        recipeIngredients = recipes.iloc[i, 1:].dropna().tolist() # Limpiamos los valores nulos y convertimos a lista los ingredientes de la receta
        recipesHDV.append(encodeRecipe(recipeIngredients, ingredientsHDV))

    '''También podemos hacer esto para los diferentes origenes de las recetas
    Para esto haremos el bundling de las recetas que tengan ese origen'''

    regions = recipes[0] # Obtenemos la columna 0 porque es dónde se encuentra la región
    regions = regions.values.flatten() # Transformamos la columna en una fila
    regions = pd.unique(regions) # Descartamos los valores repetidos

    regionsHDV = {}
 
    for i in range(len(regions)): # Con esto recorreremos todas las regiones
        regionsHDV[regions[i]] = encodeRegion(regions[i], recipesHDV, recipes)
    
    '''Ahora que tenemos todos estos datos podemos crear nuevas recetas y ver, por ejemplo,
    a que país corresponde la receta. Para esto seleccionamos los ingredientes que tenga
    la receta y haremos la operación bundling de sus ingredientes al igual que al crear las
    otras recetas.'''

    newRecipeIngredients = ['yogurt', 'lemon_juice', 'milk', 'wheat', 'yeast']  
    #newRecipeIngredients = ['butter','cane_molasses','wheat','vanilla','ginger','nutmeg','cinnamon','orange','egg','pumpkin']
    #newRecipeIngredients = ["wine", "butter", "lemon_peel", "chicken", "black_pepper", "cheese"]
    newRecipeHDV = encodeRecipe(newRecipeIngredients, ingredientsHDV)

    '''Una vez tenemos el HDV de la nueva receta podremos comparar, por ejemplo, con cuál se asemeja
    más de todas las que tenemos'''
    
    decodedRecipe = decodeRecipe(newRecipeHDV, recipesHDV, recipes)
    print(decodedRecipe)

    '''También podemos ver a que país se acerca más la receta en función de los ingredientes que implementa.'''

    bestSimilarity = -1
    bestRegion = ""
    for region, hdv in regionsHDV.items():
        similarity = torchhd.cosine_similarity(hdv, newRecipeHDV)
        if similarity > bestSimilarity:
            bestSimilarity = similarity
            bestRegion = region

    print(bestRegion)
    print(bestSimilarity)

    pass

if __name__ == "__main__":
    main()