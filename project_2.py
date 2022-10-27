from cmath import inf
from turtle import title
from typing import Counter
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import operator
import random

import numpy as np
from math import sqrt, sin, copysign

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

def front(self, n):
    return self.iloc[: , :n]

pd.DataFrame.front = front

no_cities = 30


df = pd.read_csv("CustDist_WHCorner.csv")
df_2 = pd.read_csv("CustOrd.csv")
df_3 = pd.read_csv("CustXY_WHCorner.csv")

df_dist = df.head(no_cities+1)                #For 10 cities
df_dist = df_dist.front(no_cities+2)               #For 10 cities

df_ord = df_2.head(no_cities+1)                #For 10 cities
df_ord = df_ord.front(no_cities+2)               #For 10 cities

df_xy = df_3.head(no_cities+1)                #For 10 cities
df_xy = df_xy.front(no_cities+2)               #For 10 cities


##Dataframe to np.arrays
dist = pd.DataFrame(df_dist).to_numpy()
dist = np.delete(dist,0,1)

order = pd.DataFrame(df_ord).to_numpy()


coord_xy = pd.DataFrame(df_xy).to_numpy()
coord_xy = np.delete(coord_xy,0,1)

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

IND_SIZE = no_cities

toolbox.register("indices",random.sample, range(0,IND_SIZE), IND_SIZE)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selTournament, tournsize=4)
toolbox.register("select_post", tools.selTournament, tournsize=2)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1/no_cities)
toolbox.register("mate", tools.cxOrdered)

hof = tools.HallOfFame(30)
hof_2 = tools.HallOfFame(40)

def BackBase(individual):
    sum_distances = 0
    count_orders = 0
    #back_base = np.zeros((no_cities+1,))  #Will tell the system if salesman return to the warehouse
    back_base = np.zeros((no_cities+1,))

    for i in range(0,no_cities):
        if(i == 0):
            sum_distances = dist[0][individual[0]]
            count_orders = order[individual[0]][1]
        else:
            count_orders = count_orders+order[individual[i]][1]
            if(count_orders>1000):
                back_base[individual[i-1]]=1
                back_base[individual[i]]=2
                count_orders = order[individual[i]][1]
                sum_distances = sum_distances+dist[individual[i-1]][0]
                sum_distances = sum_distances+dist[individual[i]][0] 
            else:
                sum_distances = sum_distances+dist[individual[i-1]][individual[i]]

    sum_distances = sum_distances + dist[0][individual[no_cities-1]]
    return back_base


def count_total_orders(order):
    sum_order = np.sum(order)
    return sum_order

def eval_Distances(individual):
    sum_distances = 0

    for i in range(0,no_cities):
        if(i == 0):
            sum_distances = dist[0][individual[0]+1]
        else:
            sum_distances = sum_distances+dist[individual[i-1]+1][individual[i]+1]

    sum_distances = sum_distances + dist[0][individual[no_cities-1]+1]
    return [sum_distances,]

def eval_DistancesOrder(individual):
    sum_distances = 0
    count_orders = 0
    #back_base = np.zeros((no_cities+1,))  #Will tell the system if salesman return to the warehouse
    

    for i in range(0,no_cities):
        if(i == 0):
            sum_distances = dist[0][individual[0]+1]
            count_orders = order[individual[0]+1][1]
        else:
            count_orders = count_orders+order[individual[i]+1][1]
            if(count_orders>1000):
                count_orders = order[individual[i]+1][1]
                sum_distances = sum_distances+dist[individual[i-1]+1][0]
                sum_distances = sum_distances+dist[individual[i]+1][0] 
            else:
                sum_distances = sum_distances+dist[individual[i-1]+1][individual[i]+1]

    sum_distances = sum_distances + dist[0][individual[no_cities-1]+1]
    return [sum_distances,]

toolbox.register("evaluate", eval_Distances)
toolbox.register("evaluate_order", eval_DistancesOrder)

def plot_values(best_ind_original, back_base):
    for cnt in range(no_cities):
        index = best_ind_original[cnt]

        if(cnt == 0):
            x_values = [coord_xy[0][0], coord_xy[index][0]]
            y_values = [coord_xy[0][1], coord_xy[index][1]]
            prev_index = index
            plt.plot(x_values, y_values, 'bo', linestyle='--')
            plt.text(coord_xy[0][0]-1, coord_xy[0][1] - 6, '0')
            plt.text(coord_xy[index][0]-1, coord_xy[index][1] - 6, f'{index}')
        else:
            if(back_base[index]==0):
                x_values = [coord_xy[prev_index][0], coord_xy[index][0]]
                y_values = [coord_xy[prev_index][1], coord_xy[index][1]]
                prev_index = index
                plt.plot(x_values, y_values, 'bo', linestyle='--')
                plt.text(coord_xy[index][0]-1, coord_xy[index][1] - 6, f'{index}')
            elif(back_base[index]==1):
                x_values = [coord_xy[0][0], coord_xy[index][0]]
                y_values = [coord_xy[0][1], coord_xy[index][1]]               
                plt.plot(x_values, y_values, 'bo', linestyle='--', color='red')
                x_values = [coord_xy[prev_index][0], coord_xy[index][0]]
                y_values = [coord_xy[prev_index][1], coord_xy[index][1]]
                plt.plot(x_values, y_values, 'bo', linestyle='--')
                plt.text(coord_xy[index][0]-1, coord_xy[index][1] - 6, f'{index}')
            elif(back_base[index]==2):
                x_values = [coord_xy[0][0], coord_xy[index][0]]
                y_values = [coord_xy[0][1], coord_xy[index][1]] 
                plt.plot(x_values, y_values, 'bo', linestyle='--', color='green')
                prev_index = index
                plt.text(coord_xy[index][0]-1, coord_xy[index][1] - 6, f'{index}')

        if(cnt == no_cities-1):
            x_values = [coord_xy[0][0], coord_xy[index][0]]
            y_values = [coord_xy[0][1], coord_xy[index][1]]
            plt.plot(x_values, y_values, 'bo', linestyle='--')

    plt.grid()
    plt.show()

def plot_min_average(arr_mean, arr_min):
    x_coordinate = [i+1 for i in range(len(arr_mean))]
    plt.plot(x_coordinate, arr_mean, label = "Average")
    plt.plot(x_coordinate, arr_min, label = "Min")
    plt.title("Min and Average Fitness over generations")
    plt.legend()
    plt.grid()
        

def main():
    no_population = 200

    random.seed(45)
    pop = toolbox.population(n=no_population)


    
    CXPB, MUTPB = 0.7, 0.2
    
    print("Start of evolution")

    mode = 1  #Simple: 0; Complex:1

    sum_order = count_total_orders(order)

    if(sum_order<1000):
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
    else:
        fitnesses = list(map(toolbox.evaluate_order, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]


    # Variable keeping track of the number of generations
    g = 0

    std = inf

    arr_mean = []
    arr_min = []

    max_iter = 10000/no_population
    min_total = inf
    
    while g < max_iter:
        g = g + 1
        print("-- Generation %i --" % g)

        if(g > max_iter/2 and g <= 3*max_iter/4):
            CXPB = 0.6
            MUTPB = 0.5
        elif(g > 3*max_iter/4):
            MUTPB = 0.4
        if(g >= max_iter/4 and g <= max_iter/2):
            CXPB = 0.2
            MUTPB = 0.6
        else:
            CXPB = 0.6
            MUTPB = 0.6

           
        hof.update(pop)
        hof_2.update(pop)

        hof_size = len(hof.items) if hof.items else 0
        hof2_size = len(hof_2.items) if hof_2.items else 0
        
        # Select the next generation individuals
        if(g < max_iter):
            offspring = toolbox.select(pop, len(pop)-hof_size)
            offspring.extend(hof.items)
        else:
            offspring = toolbox.select_post(pop, len(pop)-hof2_size)
            offspring.extend(hof_2.items)
        # Clone the selected individuals

        
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:

                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

    
        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

        if(min_total>min(fits)):
            min_total=min(fits)


        best_ind = tools.selBest(pop, 1)[0]
        best_ind_original = [x+1 for x in best_ind]

        print("Best individual in Generation %d %s, %s" % (g, best_ind_original, best_ind.fitness.values))

        arr_mean.append(mean)
        arr_min.append(min(fits))

    back_base = BackBase(best_ind_original)
    plot_min_average(arr_mean, arr_min)
    plt.figure()
    plot_values(best_ind_original, back_base)

if __name__ == "__main__":
    main()