import torch
import torchhd
from Bio import SeqIO
import numpy as np

vectorSize = 10000
#pathHumanData = "data/human.fasta"
#pathYeastData = "data/yeast.fasta"
pathHumanData = "/home/victor/hdv/hdvExamples/data/human.fasta"
pathYeastData = "/home/victor/hdv/hdvExamples/data/yeast.fasta"

def readFasta(path):
    headers = []
    sequences = []
    
    for record in SeqIO.parse(path, "fasta"):
        if record.description.startswith("sp|") and len(record.seq) > 2: # Nos aseguramos la longitud minima de un trímero
            headers.append(record.description)  # Descripción del registro (cabecera)
            sequences.append(str(record.seq))  # Secuencia (en forma de cadena de texto)
    
    return headers, sequences

def sequencesToHDV(sequences, trimersHDV):
    hdvs = {}

    for i, seq in enumerate(sequences):
        v = trimersHDV[seq[:3]]
        for j in range(1, len(seq) - 2):
            trimer = seq[j:j+3]
            v = torchhd.bundle(v, trimersHDV[trimer])
        hdvs[seq] = torch.sign(v)
    
    return hdvs

def predictSequence1(prototypeHuman, prototypeYeast, sequence):
    similarityHuman = torchhd.cosine_similarity(prototypeHuman, sequence)
    similarityYeast = torchhd.cosine_similarity(prototypeYeast, sequence)

    if similarityHuman > similarityYeast:
        return "human"
    else:
        return "yeast"

def predictSequence2(differenceVector, sequence):
    similarity = torchhd.cosine_similarity(differenceVector, sequence)

    if similarity > 0:
        return "human"
    else:
        return "yeast"
    
def probHDV(hdv):
    count = 0
    result = []
    hdv = torchhd.tensors.map.MAPTensor(hdv)
    
    for i in range(len(hdv)):
        if hdv[0][i] == 1:
            count += 1
    result.append(count/vectorSize)
    
    return result

def predictSequence3(theta, sequence):

    if torch.dot(theta, sequence) > 0:
        return "human"
    else:
        return "yeast"
    
def main():
    
    '''Comenzamos definiendo los aminoácidos con los que se van a trabajar.
    Para cada uno crearemos un HDV'''

    aminoacids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','U','V','W','Y']
    aminoacidsHDV = {}

    for i in range(len(aminoacids)):
        aminoacidsHDV[aminoacids[i]] = torchhd.random(1, vectorSize, "MAP")

    '''En lugar de procesar cada aminoácido individualmente los agruparemos
    en fragmentos de tres aminoácidos consecutivos (trímeros). Esto puede ser
    útil para el modelado de proteínas o análisis funcional.
    
    Para estos trímeros es importante el orden en el que se encuentras. La operación
    de binding es conmutativa por lo que deberemos usar la operación se shifting que
    altera el HDV pero conserva la información. Con esta operación podremos representar
    la posición que tienen bind(x, shift(y), shift(shift(z))).
    
    Existen en total una combinación de 9261 trímeros (21x21x21) por lo que podemos
    calcularlos todos'''

    trimersHDV = {}

    for i in range(len(aminoacids)):
        for j in range(len(aminoacids)):
            for p in range(len(aminoacids)):
                aux = torchhd.bind(aminoacidsHDV[aminoacids[i]], torchhd.permute(aminoacidsHDV[aminoacids[j]]))
                trimersHDV[aminoacids[i]+aminoacids[j]+aminoacids[p]] = torchhd.bind(aux, torchhd.permute(aminoacidsHDV[aminoacids[p]], shifts=2))
    
    '''
    A continuación leeremos un par de ficheros los cuales contienen diferentes secuencias
    de cadenas humanas y de cadenas de la levadura de cocina. Tras la lectura crearemos
    un HDV para cada secuencia a partir del bundling de los diferentes trímeros que 
    componen la secuencia.
    '''
    __, sequencesHuman = readFasta(pathHumanData)
    __, sequencesYeast = readFasta(pathYeastData)
    sequencesHumanHDV = sequencesToHDV(sequencesHuman, trimersHDV)
    sequencesYeastHDV = sequencesToHDV(sequencesYeast, trimersHDV)

    '''
    Una vez tenemos todo codificado en HDV podemos usar el 80% de los datos para el proceso
    de entrenamiento y el 20% para el proceso de testing.

    Lo primero que haremos será juntas ambos datasets de secuencias. Esto lo hacemos creando dos vectores,
    uno contendrá los HDV y otro las etiquetas que indican si es una secuencia humana o de levadura.
    Lo siguiente será mezclarlas par que los datasets tengan muestras de ambas secuencias. Después lo dividimos
    en datasets diferentes, uno para el entrenamiento y otro para el testing.
    '''

    sequencesHDV = list(sequencesHumanHDV.values()) + list(sequencesYeastHDV.values())  # Las representaciones HDV
    labels = ['human'] * len(sequencesHumanHDV) + ['yeast'] * len(sequencesYeastHDV)  # Las etiquetas correspondientes

    # Paso 2: Mezclar aleatoriamente las secuencias HDV y sus etiquetas
    indexes = np.random.permutation(len(sequencesHDV))  # Genera un arreglo de índices aleatorios
    sequencesShuffled = np.array(sequencesHDV)[indexes]  # Reordena las secuencias HDV
    labelsShuffled = np.array(labels)[indexes]  # Reordena las etiquetas

    # Paso 3: Dividir en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    trainSize = int(0.8 * len(sequencesShuffled))  # 80% de las secuencias para entrenamiento

    # Dividir las secuencias HDV y etiquetas en conjuntos de entrenamiento y prueba
    trainSequencesHDV = sequencesShuffled[:trainSize]
    testSequencesHDV = sequencesShuffled[trainSize:]

    trainLabels = labelsShuffled[:trainSize]
    testLabels = labelsShuffled[trainSize:]

    '''
    En este ejemplo veremos 3 estrategias diferentes para entrenar a un clasificador de proteinas. La primera
    consiste en crear 2 prototipos, uno para cada tipo de proteína. Cada prototipo contendra la información 
    del dataset de entrenamiento de las secuencias humanas y de levadura. Para esto se utilizará 
    la función de bundling con los HDV del dataset de entrenamiento.
    '''

    prototypeHuman = None
    prototypeYeast = None

    for i in range(len(trainSequencesHDV)):
        if trainLabels[i] == "human" and prototypeHuman is None:
            prototypeHuman = trainSequencesHDV[i]
        elif trainLabels[i] == "human":
            prototypeHuman = torchhd.bundle(prototypeHuman, trainSequencesHDV[i])
        elif trainLabels[i] == "yeast" and prototypeYeast is None:
            prototypeYeast = trainSequencesHDV[i]
        else:
            prototypeYeast = torchhd.bundle(prototypeYeast, trainSequencesHDV[i])

    prototypeHumanNoNorm = prototypeHuman # Será usado más adelante. Lo guardamos ahora para no recalcular
    prototypeYeastNoNorm = prototypeYeast # Será usad más adelante. Lo guardamos ahora para no recalcular

    prototypeHuman = torch.sign(prototypeHuman)
    prototypeYeast = torch.sign(prototypeYeast)

    '''
    Una vez tenemos los prototipos pasaremos a calcular la similitud del coseno entre la secuencia 
    que queramos clasificar y la secuencia prototipo. 
    Haremos esto para todo el dataset de testing y calcularemos lo bien que lo ha hecho.
    '''
    
    predictions = []

    for i in range(len(testSequencesHDV)):
        predictions.append(predictSequence1(prototypeHuman, prototypeYeast, testSequencesHDV[i]))

    score = 0

    for i in range(len(testLabels)):
        if testLabels[i] == predictions[i]:
            score += 1

    print("Accuracy: ")
    print(score / len(testLabels))

    '''
    La segunda estrategia consiste en extraer del dataset las que sean de levadura y obtener un vector de diferencias.
    Ahora en vez de usar los vectores prototipo utilizaremos este vector de diferencias para calcular la similitud del coseno
    '''

    differenceVector = prototypeHumanNoNorm - prototypeYeastNoNorm

    predictions = []

    for i in range(len(testSequencesHDV)):
        predictions.append(predictSequence2(differenceVector, testSequencesHDV[i]))

    score = 0

    for i in range(len(testLabels)):
        if testLabels[i] == predictions[i]:
            score += 1

    print("Accuracy: ")
    print(score / len(testLabels))

    '''
    Para la tercera y última estrategia se aplicará el modelo de Bayes. Se calculará la propabilidad de
    cada HDV humano y de levadura de que tengan el valor 1. Tras tener los dos vectores de probabilidades
    estos convergerán en uno y este será usado para las predicciones. La predicción consistirá en comprobar
    si el producto escalar entre el nuevo vector y el comparado es mayor que 0
    '''
    
    trainHumanProbs = []
    trainYeastProbs = []

    for i in range(len(testLabels)):
        if testLabels[i] == "human":
            trainHumanProbs.append(probHDV(trainSequencesHDV[i]))
        else:
            trainYeastProbs.append(probHDV(trainSequencesHDV[i]))
    
    trainHumanProbs = torch.tensor(trainHumanProbs) # Convertimos a tensores porque para realizar la operacion .log
    trainYeastProbs = torch.tensor(trainYeastProbs) # es necesario que sea un tensor

    if len(trainHumanProbs) > len(trainYeastProbs): # Tienen que tener el mismo tamaño para aplicar la resta en el calculo de theta
        trainHumanProbs[:len(trainYeastProbs)]
    elif len(trainHumanProbs) < len(trainYeastProbs):
        trainYeastProbs[:len(trainHumanProbs)]
                                                                 
    theta = (torch.log(trainHumanProbs) - torch.log(1 - trainHumanProbs)) - (torch.log(trainYeastProbs) - torch.log(1 - trainYeastProbs))

    predictions = []

    for i in range(len(testSequencesHDV)):
        predictions.append(predictSequence3(theta, testSequencesHDV[i]))

    score = 0

    for i in range(len(testLabels)):
        if testLabels[i] == predictions[i]:
            score += 1

    print("Accuracy: ")
    print(score / len(testLabels))

    pass

if __name__ == "__main__":
    main()