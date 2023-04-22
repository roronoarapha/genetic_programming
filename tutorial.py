import numpy 
import func
import matplotlib.pyplot as plt 

# equation_inputs = [4,-2,3.5,5,-11,-4.7]

# # Number of the weights we are looking to optimize.
# num_weights = len(equation_inputs)

# """
# Genetic algorithm parameters:
#     Mating pool size
#     Population size
# """
# sol_per_pop = 8
# num_parents_mating = 4

# # Defining the population size.
# pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
# #Creating the initial population.
# new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)
# print(new_population)

# """
# new_population[0, :] = [2.4,  0.7, 8, -2,   5,   1.1]
# new_population[1, :] = [-0.4, 2.7, 5, -1,   7,   0.1]
# new_population[2, :] = [-1,   2,   2, -3,   2,   0.9]
# new_population[3, :] = [4,    7,   12, 6.1, 1.4, -4]
# new_population[4, :] = [3.1,  4,   0,  2.4, 4.8,  0]
# new_population[5, :] = [-2,   3,   -7, 6,   3,    3]
# """

# best_outputs = []
# num_generations = 1000
# for generation in range(num_generations):
#     print("Generation : ", generation)
#     # Measuring the fitness of each chromosome in the population.
#     fitness = func.cal_pop_fitness(equation_inputs, new_population)
#     print("Fitness")
#     print(fitness)

#     best_outputs.append(numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))
#     # The best result in the current iteration.
#     print("Best result : ", numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))
    
#     # Selecting the best parents in the population for mating.
#     parents = func.select_mating_pool(new_population, fitness, 
#                                       num_parents_mating)
#     print("Parents")
#     print(parents)

#     # Generating next generation using crossover.
#     offspring_crossover = func.crossover(parents,
#                                        offspring_size=(pop_size[0]-parents.shape[0], num_weights))
#     print("Crossover")
#     print(offspring_crossover)

#     # Adding some variations to the offspring using mutation.
#     offspring_mutation = func.mutation(offspring_crossover, num_mutations=2)
#     print("Mutation")
#     print(offspring_mutation)

#     # Creating the new population based on the parents and offspring.
#     new_population[0:parents.shape[0], :] = parents
#     new_population[parents.shape[0]:, :] = offspring_mutation
    
# # Getting the best solution after iterating finishing all generations.
# #At first, the fitness is calculated for each solution in the final generation.
# fitness = func.cal_pop_fitness(equation_inputs, new_population)
# # Then return the index of that solution corresponding to the best fitness.
# best_match_idx = numpy.where(fitness == numpy.max(fitness))

# print("Best solution : ", new_population[best_match_idx, :])
# print("Best solution fitness : ", fitness[best_match_idx])


# import matplotlib.pyplot
# matplotlib.pyplot.plot(best_outputs)
# matplotlib.pyplot.xlabel("Iteration")
# matplotlib.pyplot.ylabel("Fitness")




"""
We define y=5+4x+3x² for x=[-10,10]
"""

# Inputs of the equation.
x=numpy.linspace(-1,1,1000)
y=5+4*x-5*x**2+6*x**3

num_weights = 4

sol_per_pop = 32
num_parents_mating = 4

pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
new_population = numpy.random.uniform(low=-10.0, high=10.0, size=pop_size)

best_outputs = []
num_generations = 200
print ("la pop initiale vaut", new_population)
for generation in range(num_generations):
    plt.clf()

    # print("Generation : ", generation)
    # Measuring the fitness of the function parameters.
    finesse = func.cal_pop_fitness(x, y, new_population)
    # print("Fitness")
    # print(finesse)
    best_outputs.append(min(finesse))
    # The best result in the current iteration.
    # print("Best result : ", min(finesse))
    # Selecting the best parents in the population for mating.
    parents = func.select_mating_pool(new_population, finesse, 
                                      num_parents_mating)
    # print("Parents")
    # print(parents)

    # Generating next generation using crossover.
    offspring_size=(pop_size[0]-parents.shape[0], num_weights)
    # print("offspring =", offspring_size)

    offspring_crossover = func.crossover(parents, offspring_size)
    # print("Crossover")
    # print(offspring_crossover)

    # Adding some variations to the offspring using mutation.
    
    offspring_mutation = func.mutation(offspring_crossover, num_mutations=3)
    # print("Mutation")
    # print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    # print("new parent 1", new_population)
    new_population[parents.shape[0]:, :] = offspring_mutation
    # print("new parent 2", new_population)

    finesse = func.cal_pop_fitness(x,y, new_population)

    best_match_idx = finesse.index(min(finesse))

    # print("Best solution : ", new_population[best_match_idx, :])
    # print("Best solution fitness : ", finesse[best_match_idx])
    b_opti=new_population[best_match_idx, :]
    y_opti=b_opti[0]+b_opti[1]*x+b_opti[2]*x**2+b_opti[3]*x**3
    plt.subplot(2, 1, 1)
    plt.plot(x,y,label="original")
    plt.plot(x,y_opti,label="estimé")
    plt.grid()
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.grid()
    plt.plot(best_outputs,label="finesse")
    plt.legend()
    plt.show(block=False)
    plt.pause(0.00001)
    
finesse = func.cal_pop_fitness(x,y, new_population)

best_match_idx = finesse.index(min(finesse))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", finesse[best_match_idx])
plt.show()

# matplotlib.pyplot.show()