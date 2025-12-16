import torch
import torchhd
import matplotlib.pyplot as plt
import time
import psutil

vectorSize = 10000
colorPrecision = 4
sampleSize = 1000

def randomColor():
    colorList = torch.rand(3).tolist()
    return [round(num, colorPrecision) for num in colorList]

def rangeHDVs(numSteps):
    k = numSteps - 1
    V = torchhd.random(numSteps, vectorSize, "MAP")

    for i in range(1, numSteps):
        for j in range(vectorSize):
            V[i][j] = torch.where(torch.rand(1) < (1/k), -V[i-1][j], V[i-1][j])
        
    return V
            
def colorIndex(v, numSteps):
    return int(round(v * (numSteps - 1), 1))

def encodeColor(color, redHDV, greenHDV, blueHDV, numSteps):
    '''
    La variable "color" debe de ser un vector de 3 componentes.
    Cada componente debe de ser un valor entre 0 y 1
        Posiciones:
            Rojo:   0
            Verde:  1
            Azul:   2
    '''
    b1 = torchhd.bind(redHDV[colorIndex(color[0], numSteps)], greenHDV[colorIndex(color[1], numSteps)])
    return torchhd.bind(b1, blueHDV[colorIndex(color[2], numSteps)]) 

def decodeColor(colorHDV, colorSample):

    bestSimilarColor = ""
    bestSimilarity = -1

    for color, sample in colorSample.items():
        similarity = torchhd.cosine_similarity(colorHDV, sample)
        if similarity > bestSimilarity:
            bestSimilarity = similarity
            bestSimilarColor = color
    print(bestSimilarity)
    return bestSimilarColor

def main():
    gota, sol, huevo, lobo, camion, planta, platano = torchhd.random(7, vectorSize,"MAP")

    '''
    Para representar números en un intervalo fijo [a,b]
    Se ha de dividir ese intervalo en k partes iguales

    Tras la división se genera un HDV para representar la parte infrerior del intervalo
    Se reemplazara 1/k del anterior vector con cada paso

    '''

    colorSteps = 21 # 21 rangos debido a que se quiere representar intervalos de 0.05

    redHDV = rangeHDVs(colorSteps)
    greenHDV = rangeHDVs(colorSteps)
    blueHDV = rangeHDVs(colorSteps)

    '''
    Cada color viene representado por una matriz la cual representa
    los diferentes intervalos de un color primario.

    Para crear otro tipo de color de hará la operación de binding
    de los 3 canales que hemos creado.
    '''

    color = [0.3229, 0.4557, 0.1225]
    colorHDV = encodeColor(color, redHDV, greenHDV, blueHDV, colorSteps)

    '''
    Una vez sabemos codificar los colores necesitamos una muestra.
    Con esta muestra podremos comparar los diferentes colores para ver a cuáles se asemeja más
    '''
    colorSample = {}
    for i in range(0, sampleSize): #Así creamos 1000 colores diferentes que usaremos como muestra
        randColor = randomColor()
        colorSample[tuple(randColor)] = encodeColor(randColor, redHDV, greenHDV, blueHDV, colorSteps)

    #colorSample[(0.320, 0.450, 0.001)] = colorHDV
    '''
    También podemos, dado un HDV que representa un vector, obtener MÁS O MENOS el color con el que se corresponde.
    Lo bien que se haga esto dependerá de la variedad de la muestra debido a que el proceso consistirá en
    buscar en la muestra el color que más similitud tenga con el dado.
    Esto lo hará nuestra función decodeColor()
    '''
    process = psutil.Process()
    cpuStart = process.cpu_percent(interval=1)
    memoryStart = process.memory_info().rss / (1024*1024) #mb
    start = time.time()

    color = [0.3229, 0.4557, 0.1225]
    colorHDV = encodeColor(color, redHDV, greenHDV, blueHDV, colorSteps)
    colorDecoded = decodeColor(colorHDV, colorSample)
    
    end = time.time()
    cpuEnd = process.cpu_percent(interval=1)
    memoryEnd = process.memory_info().rss / (1024*1024) #mb
   
    print(color)
    print(colorDecoded)
    print(f"Timepo (Segundos): {end-start}")
    print(f"Uso de CPU antes: {cpuStart}%")
    print(f"Uso de CPU después: {cpuEnd}%")
    print(f"Uso de memoria antes: {memoryStart:.2f} MB")
    print(f"Uso de memoria después: {memoryEnd:.2f} MB")

    # Crear la figura y mostrar los colores
    
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
    ax[0].set_title('Color 1')
    ax[0].axis('off')

    ax[1].add_patch(plt.Rectangle((0, 0), 1, 1, color=colorDecoded))
    ax[1].set_title('Color 2')
    ax[1].axis('off')
    plt.show()
    
    pass

if __name__ == "__main__":
    main()