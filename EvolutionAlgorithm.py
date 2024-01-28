from visualization import visualizeMap, make_plot, make_histogram
import numpy as np
import random
import math
import copy
import uuid
import regex as re
from collections import defaultdict
from ParseXML import get_root, get_nodes, get_links, get_demands_with_values
import argparse
from utils import save_data

def calculate_fitness(ind, demands_paths, m, recip=False):
    edges_dict = defaultdict(lambda:0)
    fitness = 0
    for link, paths_dict in ind.items():
        for path, edge_value  in paths_dict.items():
            demand = 'Demand_'+link.partition('_')[-1]
            for direct_path in demands_paths[demand][path]:
                edges_dict[direct_path]+=edge_value
    
    for _, value in edges_dict.items():
        fitness+=math.ceil(value/m)

    if recip == False:
        return 1/fitness 
    else: return fitness

def initialize_population(pop_size, dissagregation, demands_paths, demands_values):
    pop = []

    for ind in range(pop_size):
        individual = defaultdict(lambda: defaultdict(int))
        for idx, ((d_paths_key, d_paths_val), (d_values_key, d_values_val)) in enumerate(zip(demands_paths.items(), demands_values.items())):
            link_id = 'Link_'+d_paths_key.partition('_')[-1]

            if dissagregation:
                num_of_paths = random.randint(1, len(list(d_paths_val.keys())))

                sample_paths = random.sample(list(d_paths_val.keys()), k=num_of_paths)
                
                weights = np.random.multinomial(d_values_val, [1/num_of_paths]*num_of_paths, size=1)
                for path, weight in zip(sample_paths, *weights):
                    individual[link_id][path] = weight

            else:
                num_of_paths = 1

                sample_paths = random.sample(list(d_paths_val.keys()), k=num_of_paths)
                individual[link_id][str(*sample_paths)] = d_values_val
            
        pop.append(individual)

    return pop 

def roulette_wheel(pop, demand_paths, m):
    pop_size = len(pop)
    pop_after_preselection = []

    fitness_list = []
    for ind in pop:
        fitness_list.append(calculate_fitness(ind, demand_paths, m))

    fitness_sum = sum(fitness_list)
    probabilities = []

    for i in range(pop_size):
        probabilities.append(fitness_list[i]/fitness_sum)

    # Generate probability intervals for each individual
    intervals = [sum(probabilities[:i+1]) for i in range(len(probabilities))]

    while len(pop_after_preselection)<pop_size:
        p_s = random.uniform(0, 1)
        for idx, ind in enumerate(pop):
            if p_s <= intervals[idx]:
                pop_after_preselection.append(ind)
                break
    return pop_after_preselection

def tournament_selection(pop, demands_paths, tournament_size, m):
    '''
        Fitness calculated inside of the preselection
    '''
    pop_after_tournament = []

    for i in range(len(pop)):
        lists = random.sample(pop, k=tournament_size)
        fit_vals = []
        for ind in lists:
            fit = calculate_fitness(ind, demands_paths, m)
            fit_vals.append(fit)
        idx = max(enumerate(fit_vals), key=lambda x: x[1])[0]
        pop_after_tournament.append(lists[idx])

    return pop_after_tournament

def uniform_mutation(pop, demands_paths, demands_values, mut_rate, dissagregation):
    for _ in range(len(pop)):
        l = random.randint(0, len(pop)-1)
        if random.uniform(0, 1) < mut_rate:
            k = random.randint(0, len(pop)-1)

            for idx, ((d_paths_key, d_paths_val), (d_values_key, d_values_val)) in enumerate(zip(demands_paths.items(), demands_values.items())):
                if idx == k:
                    link_id = 'Link_'+d_paths_key.partition('_')[-1] # change from demand_# to link_#

                    if dissagregation:
                        num_of_paths = random.randint(1, len(list(d_paths_val.keys())))
                        sample_paths = random.sample(list(d_paths_val.keys()), k=num_of_paths)        
                        weights = np.random.multinomial(d_values_val, [1/num_of_paths]*num_of_paths, size=1)
                        for path, weight in zip(sample_paths, *weights):
                            pop[k][link_id][path] = weight

                    else:
                        num_of_paths = 1
                        sample_paths = random.sample(list(d_paths_val.keys()), k=num_of_paths)
                        pop[k][link_id][str(*sample_paths)] = d_values_val
    return pop

def uniform_crossover(pop, cross_rate):
    pop_after_crossover = []

    while len(pop_after_crossover) <= len(pop):
        individual = defaultdict(lambda: defaultdict(int))
        parents = random.sample(pop, k=2)
        list_of_links = list(parents[0].keys())
        list_of_paths = list(parents[0].values())
        list_of_links_other = list(parents[1].keys())
        list_of_paths_other = list(parents[1].values())

        for i in range(0, len(pop[0])):
            if random.uniform(0, 1) > cross_rate:
                individual[list_of_links[i]] = list_of_paths[i]
            else:
                individual[list_of_links_other[i]] = list_of_paths_other[i]

        pop_after_crossover.append(individual)

    return pop_after_crossover

def two_points_crossover(pop):
    pop_after_crossover = []

    def get_random_indices(limit, size):
        indices = set()
        answerSize = 0

        while answerSize < size:
            r = random.randint(0,limit-1)
            if r not in indices:
                answerSize += 1
                indices.add(r)

        return min(indices), max(indices)

    while len(pop_after_crossover) < len(pop):
        individual = defaultdict(lambda: defaultdict(int))
        parents = random.sample(pop, k=2)
        list_of_links = list(parents[0].keys())
        list_of_paths = list(parents[0].values())
        list_of_links_other = list(parents[1].keys())
        list_of_paths_other = list(parents[1].values())

        left, right = get_random_indices(len(pop[0]), 2)

        for i in range(0, len(pop[0])):
            if i<left:
                individual[list_of_links_other[i]] = list_of_paths_other[i]
            elif i>=right:
                individual[list_of_links[i]] = list_of_paths[i]
            else:
                individual[list_of_links[i]] = list_of_paths[i]
    
        pop_after_crossover.append(individual)

    return pop_after_crossover

def elite_succession(pop, old_pop, nodes, links, demands_paths, demands_values, elite_size, m):
    pop_merged = pop

    fitness_old = []
    for ind in old_pop:
        fitness_old.append(calculate_fitness(ind, demands_paths, m))
    
    # sorting simultaneously so as not to lose information about indexes of certain chromosomes

    indexes = sorted(range(len(fitness_old)), key=fitness_old.__getitem__, reverse=True)

    old_pop = list(map(old_pop.__getitem__, indexes))
    old_pop = old_pop[:elite_size]

    for i in range(len(old_pop)):
        pop_merged[i] = old_pop[i]

    fitness_merged = []
    for ind in pop_merged:
        fitness_merged.append(calculate_fitness(ind, demands_paths, m))

    return pop_merged

def succession_with_partial_replacement(pop, old_pop, demands_paths, m):
    pop_merged = pop + old_pop
    fitness_merged = []
    for ind_merged in pop_merged:
        fitness_merged.append(calculate_fitness(ind_merged, demands_paths, m))

    indexes = sorted(range(len(fitness_merged)), key=fitness_merged.__getitem__, reverse=True)

    pop_merged = list(map(pop_merged.__getitem__, indexes))
    pop_merged = pop_merged[:len(pop)]

    return pop_merged

def start_evolutionary_algorithm(msrmnt_id, type_, dissagregation, pop_size, nodes, links, demands_paths, demands_values, num_of_iterations, tournament_size, mut_rate, cross_rate, elite_size, m, no_plots=False):
    pop = initialize_population(pop_size=pop_size, dissagregation=dissagregation, demands_paths=demands_paths, demands_values=demands_values)
    unique_filename = str(uuid.uuid4())
    avgs = []
    maxis = []
    minis = []
    history = []
    fitness_recip_of_last_max = 0
    
    fitness_merged = []
    fitness_recip_merged = []
    for ind in pop:
        fitness_merged.append(calculate_fitness(ind, demands_paths, m))
        fitness_recip_merged.append(calculate_fitness(ind, demands_paths, m, True))
    print(f'Init', ' Avg: ', sum(fitness_merged)/len(fitness_merged), ' Max: ', max(fitness_merged), ' Min: ', min(fitness_merged))

    if type_== 1:
        '''
            1. roulette preselection
            2. uniform mutation
            3. two-points crossover
            4. succession with partial replacement
        '''

        history.append((pop, fitness_recip_merged))

        for i in range(num_of_iterations):
            old_pop = copy.deepcopy(pop)
            pop = roulette_wheel(pop, demands_paths, m)
            pop = uniform_mutation(pop, demands_paths, demands_values, mut_rate, dissagregation)
            pop = two_points_crossover(pop)
            pop = succession_with_partial_replacement(pop, old_pop, demands_paths, m)

            fitness_merged = []
            fitness_recip_merged = []
            for ind in pop:
                fitness_merged.append(calculate_fitness(ind, demands_paths, m))
                fitness_recip_merged.append(calculate_fitness(ind, demands_paths, m, True))
            history.append((pop, fitness_recip_merged))
            avg = sum(fitness_merged)/len(fitness_merged)
            avgs.append(avg)
            maxi = max(fitness_merged)
            maxis.append(maxi)
            mini = min(fitness_merged)
            minis.append(mini)
            # print(f'Iteration: {i+1}', ' Avg: ', avg, ' Max: ', maxi, ' Min: ', mini)
            save_data([i+1, avg, maxi, mini], unique_filename)

        fitness_of_last_max = maxis[-1]
        fitness_recip_of_last_max = calculate_fitness(pop[-1], demands_paths, m, True)
        info = (dissagregation, fitness_of_last_max, fitness_recip_of_last_max, m, pop_size, num_of_iterations, mut_rate, cross_rate)

        if not no_plots:
            make_plot(avgs, maxis, minis, num_of_iterations, unique_filename, type_, info)
            visualizeMap(links, nodes, demands_paths, pop, unique_filename, type_, info)
    elif type_==2:
        '''
            1. tournament preselection
            2. uniform mutation
            3. uniform crossover
            4. elitist succession
        '''

        history.append((pop, fitness_merged))

        for i in range(num_of_iterations):
            old_pop = copy.deepcopy(pop)
            pop = tournament_selection(pop, demands_paths, tournament_size, m)
            pop = uniform_mutation(pop, demands_paths, demands_values, mut_rate, dissagregation)
            pop = uniform_crossover(pop, cross_rate)
            pop = elite_succession(pop, old_pop, nodes, links, demands_paths, demands_values, elite_size, m)

            fitness_merged = []
            fitness_recip_merged = []
            for ind in pop:
                fitness_merged.append(calculate_fitness(ind, demands_paths, m))
                fitness_recip_merged.append(calculate_fitness(ind, demands_paths, m, True))
            history.append((pop, fitness_recip_merged))
            avg = sum(fitness_merged)/len(fitness_merged)
            avgs.append(avg)
            maxi = max(fitness_merged)
            maxis.append(maxi)
            mini = min(fitness_merged)
            minis.append(mini)
            # print(f'Iteration: {i+1}', ' Avg: ', avg, ' Max: ', maxi, ' Min: ', mini)
            save_data([i+1, avg, maxi, mini], unique_filename)

        fitness_of_last_max = maxis[-1]
        fitness_recip_of_last_max = calculate_fitness(pop[-1], demands_paths, m, True)
        info = (dissagregation, fitness_of_last_max, fitness_recip_of_last_max, m, pop_size, num_of_iterations, tournament_size, mut_rate, cross_rate, elite_size)

        if not no_plots:
            make_plot(avgs, maxis, minis, num_of_iterations, unique_filename, type_, info)
            visualizeMap(links, nodes, demands_paths, pop, unique_filename, type_, info)
    else:
        print("There are only two types of algorithm implemented")
        return
    
    if not no_plots:
        gen = 27
        make_histogram(history[gen][1], unique_filename, gen)
        gen = 153
        make_histogram(history[gen][1], unique_filename, gen)
        gen = 273
        make_histogram(history[gen][1], unique_filename, gen)
        gen = 450
        make_histogram(history[gen][1], unique_filename, gen)

    return fitness_recip_of_last_max

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def main():
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('--dissagregation', '--d')
    parser.add_argument('--modularity', '--m', )
    parser.add_argument('--type_', '--t')  
    parser.add_argument('--measurement_id', '--mid')
    parser.add_argument( '--pop_size', '--p')
    parser.add_argument('--num_of_iterations', '--n')
    parser.add_argument('--tournament_size', '--ts')
    parser.add_argument( '--mut_rate', '--mr')
    parser.add_argument('--cross_rate', '--cr')
    parser.add_argument('--elite_size', '--es')     
    args = parser.parse_args()
    
    type_ = int(args.type_)
    mesaurement_id = int(args.measurement_id)

    root = get_root('./data/polska.xml')

    nodes = get_nodes(root)
    links = get_links(root)
    demands_paths, demands_values = get_demands_with_values(root)
    N = 66
    dissagregation = True if int(args.dissagregation == 1) else False
    m = int(args.modularity)
    pop_size = int(args.pop_size)
    num_of_iterations = int(args.num_of_iterations)
    tournament_size = int(args.tournament_size)
    mut_rate = float(args.mut_rate)
    cross_rate = float(args.cross_rate)
    elite_size = int(args.elite_size)

    start_evolutionary_algorithm(mesaurement_id, type_, dissagregation, pop_size, nodes, links, demands_paths, demands_values, num_of_iterations, tournament_size, mut_rate, cross_rate, elite_size, m)

if __name__ == "__main__":
    main()