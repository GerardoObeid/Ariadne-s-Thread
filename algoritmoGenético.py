import random
import numpy as np
from matplotlib import pyplot as plt
from copy import copy
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import pandas as pd
from IPython.display import display


class Individuo():
    def __init__(self):
        self.individual_number = 0

    def createIndividuo(self):
        individuo = []
        for i in range(20):
            magnitude = round(random.uniform(-3, 3), 3)
            direction = random.randint(0, 1)
            point = [0, 0]
            if direction == 0:
                point[0] = magnitude
                point[1] = 0
            else:
                point[1] = magnitude
                point[0] = 0
            point = tuple(point)
            individuo.append(point)
        return individuo


class populationManager():
    def __init__(self):
        self.len = 100
        self.individuos = []

    def createFirstGeneration(self):
        individuos = self.individuos
        lenPoblacion = self.len
        for i in range(lenPoblacion):
            newIndividuo = Individuo()
            ind = newIndividuo.createIndividuo()
            individuos.append(ind)
        return individuos


class AlgoritmoGenetico():
    def __init__(self, reloaded=False):
        self.exit = []
        self.population = []
        self.numGeneration = 0
        self.fitnessList = []
        self.Generations = 200000
        self.bestFitnessDict = {}
        self.numMut = 0
        self.xs = []
        self.ys = []
        self.reloaded = reloaded

    def initializeProcess(self):
        # x = int(round(random.uniform(-30, 30), 3))
        # y = int(round(random.uniform(-30, 30), 3))
        x = round(random.uniform(-30, 30), 3)
        y = round(random.uniform(-30, 30), 3)
        self.exit = [x, y]
        poblacion = populationManager()
        self.population = poblacion.createFirstGeneration()

    def continueProcessing(self):
        fitness = []
        orderedFitness = self.fitness(self.population)
        self.fitnessList.append(orderedFitness[0][1])
        fit = orderedFitness[0][1]
        self.bestFitnessDict[fit] = 1
        Generations = self.Generations
        for i in tqdm(range(Generations)):
            self.numGeneration += 1
            generation = self.cruza(orderedFitness)
            if self.bestFitnessDict[fit] >= 40:
                generation = self.mutacion(generation)

            orderedFitness = self.fitness(generation)
            self.fitnessList.append(orderedFitness[0][1])
            fitness.append(orderedFitness[0][1])

            fit = orderedFitness[0][1]
            if fit not in self.bestFitnessDict.keys():
                self.bestFitnessDict[fit] = 1
            else:
                self.bestFitnessDict[fit] += 1

            # criterios de salida
            if fit == 0:
                self.getLastBestFitness(orderedFitness)
                self.fitnessList.append(orderedFitness[0][1])
                self.fitnessList.append(orderedFitness[0][1])
                break
            # Exit case
            if ((fit < 0.0001 and i > 20000)
                    or self.bestFitnessDict[fit] > 25000):
                break
        return orderedFitness

    def graphFitness(self):
        return self.fitnessList

    def fitness(self, population):
        exit = self.exit
        data = []
        listxs = []
        listys = []
        for j in range(len(population)):
            sumX = 0
            sumY = 0
            for x, y in population[j]:
                sumX += x
                sumY += y
            sumX = round(sumX, 3)
            sumY = round(sumY, 3)
            resX = exit[0] - sumX
            resY = exit[1] - sumY
            resX = round(resX, 3)
            resY = round(resY, 3)
            magnitude = np.sqrt(resX**2 + resY**2)

            data.append(tuple([population[j], magnitude]))

            listxs.append(sumX)
            listys.append(sumY)
            
        self.xs.append(listxs)
        self.ys.append(listys)
        data.sort(key=lambda tup: tup[1])
        return data[0:100]

    def getLastBestFitness(self, generationWithFitness):
        listxs = []
        listys = []
        for j in range(len(generationWithFitness)):
            sumX = 0
            sumY = 0
            for x, y in generationWithFitness[j][0]:
                sumX += x
                sumY += y
            sumX = round(sumX, 3)
            sumY = round(sumY, 3)
            listxs.append(sumX)
            listys.append(sumY)

        self.xs.append(listxs)
        self.ys.append(listys)

    def cruza(self, orderedPopulation):
        chosenCruza = []
        newOrganismos = []
        for i in range(25):
            chosenCruza.append(orderedPopulation[i][0])
        for i in range(59, 49, -1):
            chosenCruza.append(orderedPopulation[i][0])
        for i in range(99, 94, -1):
            chosenCruza.append(orderedPopulation[i][0])

        random.shuffle(chosenCruza)

        for i in range(0, 40, 2):        
            puntoDeCorte = random.randint(4, 15)

            hijo1 = chosenCruza[i][0:puntoDeCorte]+chosenCruza[i+1][puntoDeCorte:20]
            hijo2 = chosenCruza[i+1][0:puntoDeCorte]+chosenCruza[i][puntoDeCorte:20]
            newOrganismos.append(hijo1)
            newOrganismos.append(hijo2)

        for individuo in orderedPopulation:
            newOrganismos.append(individuo[0])

        if len(newOrganismos) < 140:
            print(len(newOrganismos))
        return newOrganismos

    def mutacion(self, population):
        self.numMut += 1
        mutatedNumbers = []
        if not self.reloaded:
            mutationRate = 50
        else:
            mutationRate = 100
        for i in range(mutationRate):
            newGenes = []
            newNumber = True
            while newNumber:
                toMutate = random.randint(0, len(population)-1)
                if toMutate not in mutatedNumbers:
                    newNumber = False
                    mutatedNumbers.append(toMutate)

            mutateGenes = random.randint(1, 4)
            for x in range(mutateGenes):
                newNumber = True
                while newNumber:
                    posMutation = random.randint(0, 19)
                    if posMutation not in newGenes:
                        newNumber = False
                        newGenes.append(posMutation)

                magnitude = round(random.uniform(-3, 3), 3)
                direction = random.randint(0, 1)
                point = [0, 0]
                if direction == 0:
                    point[0] = magnitude
                    point[1] = 0
                else:
                    point[1] = magnitude
                    point[0] = 0
                point = tuple(point)

                if self.reloaded:
                    if x == 0:
                        ind = copy(population[toMutate])
                        ind[posMutation] = point
                    else:
                        ind[posMutation] = point
                else:
                    population[toMutate][posMutation] = point

            if self.reloaded:
                population.append(ind)      
        return population


def graph(listx, listy, lenlists, exit):
    xs = ys = []
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    colors = ['g']
    colorsPob = ['r']
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set(xlim=(-35, 35), ylim=(-35, 35))
    scat = ax.scatter(xs, ys, c=colorsPob, s=45)
    ax.scatter(exit[0], exit[1], c=colors, s=30)
    result = []

    def animate(frame_num):
        xs = listx[frame_num]
        ys = listy[frame_num]
        scat.set_offsets(np.c_[xs, ys]) 
        ax.set_title('Gen ' + str(frame_num))
        result.append(xs)
        return scat

    anim = FuncAnimation(fig,
                         animate,
                         frames=lenlists,
                         interval=350,
                         repeat=False)
    plt.show()


def graphMovement(bestDesplazon, exit):
    x = y = 0
    xs = [0]
    ys = [0]
    for xx, yy in bestDesplazon:
        x += xx
        y += yy
        xs.append(round(x, 3))
        ys.append(round(y, 3))

    # Select length of axes and the space between tick labels
    xmin, xmax, ymin, ymax = -30, 30, -30, 30
    ticks_frequency = 10

    fig, ax = plt.subplots(figsize=(8, 8))

    plt.scatter(exit[0], exit[1], color='red', s=200)
    plt.scatter(xs, ys, color='green', s=.1)

    ax.set(xlim=(xmin-1, xmax+1), ylim=(ymin-1, ymax+1), aspect='equal')
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('x', size=14, labelpad=-24, x=1.03)
    ax.set_ylabel('y', size=14, labelpad=-21, y=1.02, rotation=0)
    x_ticks = np.arange(xmin, xmax+1, ticks_frequency)
    y_ticks = np.arange(ymin, ymax+1, ticks_frequency)
    ax.set_xticks(x_ticks[x_ticks != 0])
    ax.set_yticks(y_ticks[y_ticks != 0])

    plt.plot(xs, ys, '-o')

    ax.set_xticks(np.arange(xmin, xmax+1), minor=True)
    ax.set_yticks(np.arange(ymin, ymax+1), minor=True)

    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

    plt.show()


def logger(AG: AlgoritmoGenetico, obtained_exit, pop, display_df=False):
    selected_fitness_list = []
    selected_fitness_list.append(round(AG.fitnessList[0], 4))
    eachGen = int((len(AG.fitnessList) / 48) + 1)

    for i in range(len(AG.fitnessList)-1):
        if i % eachGen == 0:
            selected_fitness_list.append(round(AG.fitnessList[i], 4))
    selected_fitness_list.append(round(AG.fitnessList[-1], 4))
    df = pd.DataFrame([[AG.numGeneration, AG.exit, obtained_exit, selected_fitness_list, pop]])

    # agregar valores
    df.columns = ['Number_of_generations', 'Exit_point (Original)', 'Exit_point (best movement)', 'Best_fitness (each (total / 50))', 'Hilo_de_Ariadna']
    df_final = pd.read_csv("AG_logging.csv")
    df_final = pd.concat([df_final, df], ignore_index=True, sort=False)
    df_final.to_csv("AG_logging.csv", index=False)
    # No enseña el dataFrame de una manera ilutrativa por lo que el valor de display_Df se quedará como default en falso
    if display_df:
        display(df.head())


# si se crea el objeto mandando el parámetro en True, se activa la versión reloaded que realiza el trabajo más de 3 veces mejor.
# La versión reloaded guarda las mutaciones como nuevos individuos, por esta razón es mas preciso y eficaz
def main():
    AG = AlgoritmoGenetico()
    AG.initializeProcess()
    data = AG.continueProcessing()
    pop = data[0][0]
    sumX = 0
    sumY = 0
    for x, y in pop:
        sumX += x
        sumY += y
    sumX = round(sumX, 3)
    sumY = round(sumY, 3)
    print("\nCoordenadas obtenidas: ", "("+str(sumX) + ",", str(sumY)+")")
    print("Salida verdadera:",tuple(AG.exit))
    li = AG.graphFitness()
    print("Fitness inicial:", li[0])
    print("Fitness final:", li[-1])
    # print("Desplazón Perfecto:", data[0][0])
    graph(AG.xs, AG.ys, len(AG.xs), AG.exit)
    graphMovement(data[0][0], AG.exit)
    logger(AG, [sumX, sumY], pop)


if __name__ == "__main__":
    main()
