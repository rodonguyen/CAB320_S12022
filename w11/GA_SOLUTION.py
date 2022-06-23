'''

Modify on 2021 June 15
by f.maire@qut.edu.au
added 
   Individual._onePointCrossover()
   
   
   

'''

#GA.py, QUT lecture on Evolutionary Computing
#A simple GA with missing code sections
#Solving the one-max problem


import random
import copy

import matplotlib.pyplot as plt
import numpy

TOURNAMENT_SIZE=4
POPULATION_SIZE=20

MUTATION_RATE=0.1
CROSSOVER_RATE=0.9

GENERATION_LIMIT=10000
MAXIMUM_FITNESS=30
REPORT_FREQUENCY=10

ELITISM_NUMBER=1


class Stats(object):
  def __init__(self):
    self.maxs=[]
    self.mins=[]
    self.avgs=[]

  def _recordGeneration(self):
    self.maxs.append(calculateBestFitness())
    self.mins.append(calculateWorstFitness())
    self.avgs.append(calculateAverageFitness())

class Individual(object):
  length=30 #how many bits in the genome
  genome=[] #start with an empty genome
  alleles=(0,1) #choices per-gene are 0 or 1

  def __init__(self, genome=None):
    self.genome = genome or self._createGenome()
    self.fitness = None  # set during evaluation

  #flip a single bit in the genome
  def _mutateBit(self, gene):
    if (self.genome[gene]==0):
      self.genome[gene]=1
    else:
      self.genome[gene]=0

  #select a single point in the genome and replace everything after it 
  #with the same indices from a second genome
  def _onePointCrossover(self, other):
    point = random.randrange(1, self.length)
    self.genome[point:] = other.genome[point:]


  #find a subsection of a second genome and replace the child's genome with it
  def _twoPointCrossover(self, other):
    first = random.randrange(1, self.length-2)
    second = random.randrange(first, self.length-1)

    self.genome[first:second] = other.genome[first:second]

  #populate a genome with 0s and 1s
  def _createGenome(self):
    return [random.choice(self.alleles) for gene in range(self.length)]

  def _printIndividual(self):
    print(f'fitness: {self.fitness}, genome: {self.genome}')



def randomSelect():
  return random.choice(population)

#non-class methods

#select using a tournament
def tournamentSelect(select_best_prob=1.0):
  tourney = [random.choice(population) for i in range(TOURNAMENT_SIZE)]
  tourney.sort(key=lambda x: x.fitness, reverse=True)

  if random.random() < select_best_prob:
    return tourney[0]
  else:
    return random.choice(tourney[1:])


def printExperimentStats():
 # print "="*70
  print ("generation: ", generation_counter)
  print ("best: ", calculateBestFitness() )
  print ("average: ", calculateAverageFitness())
  print ("worst: ", calculateWorstFitness(), "\n")



def calculateBestFitness():
  return max(individual.fitness for individual in population)

def calculateAverageFitness():
  return float (sum(individual.fitness for individual in population))/float(POPULATION_SIZE)

def calculateWorstFitness():
  return min(individual.fitness for individual in population)


def createChildren():
  #sort the population by fitness
  population.sort(key=lambda x: x.fitness, reverse=True)

  #If we use elitism, keep the best individuals in the next population
  next_population=copy.deepcopy(population[:ELITISM_NUMBER])

  #Now create random children up to POPULATION_SIZE
  while(len(next_population) < POPULATION_SIZE):
   # new_child = randomSelect()
    new_child = tournamentSelect()

    #perform crossover
    if(random.random() < CROSSOVER_RATE):
      new_child._twoPointCrossover(tournamentSelect())

    #perform mutation
    for i in range(len(new_child.genome)):
      if(random.random()<MUTATION_RATE):
        new_child._mutateBit(i)

    #assess fitness
    new_child.fitness = sum(new_child.genome)

    #add it to the new population
    next_population.append(copy.deepcopy(new_child))

  return next_population

#plot the best, average, and worst fitness for the experiment
def showGraphs(stats):
  fig = plt.figure()
  ax = fig.add_subplot(111)

  ax.plot(numpy.arange(0,len(stats.maxs)), stats.maxs, color="red",linewidth=2.,linestyle="-",label="Best f")
  ax.plot(numpy.arange(0,len(stats.mins)), stats.mins, color="blue",linewidth=2.,linestyle=":",label="Worst f")
  ax.plot(numpy.arange(0,len(stats.avgs)), stats.avgs, color="green",linewidth=2.,linestyle="-.",label="Avg f")

  ax.set_ylabel('Fitness values', fontsize='12')
  ax.set_xlabel('Generation', fontsize='12')

  plt.show()








######################
###main loop starts###
######################
generation_counter=0
population=[]
stats = Stats()

#create the initial population
while len(population) < POPULATION_SIZE:
  population.append(Individual())


#evaluate population and assign fitness
#for the one-max problem, we want every allelle to be 1
for indiv in population:
  indiv.fitness= sum(indiv.genome)

#print some fitness stats to screen
printExperimentStats()


#the problem is solved if we discover a genome has 1 for every allelle
solved=False
while(generation_counter<GENERATION_LIMIT and not solved):

  #evolve
  population = createChildren()
  stats._recordGeneration()

  #report on the fitness progression
  if(generation_counter%REPORT_FREQUENCY == 0):
    printExperimentStats()

  # If an individual has the maximum fitness, we have found the optimum and can stop
  if(calculateBestFitness()==MAXIMUM_FITNESS):
    solved=True
    print ("Solved, exiting! Generations: ",generation_counter)
    showGraphs(stats)

  generation_counter = generation_counter+1


