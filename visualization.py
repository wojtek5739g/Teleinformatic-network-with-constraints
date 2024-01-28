import matplotlib.pyplot as plt
from collections import defaultdict
import math

def make_plot(avgs, maxis, minis, num_of_iterations, unique, type, info):
    if type==1:
        text = f'Dissagregation: {info[0]}\nFitness of last max (solution): {round(info[1], 10)}\nModularity: {info[3]}\nUsed systems: {info[2]}\n'\
        f'Population size: {info[4]} \nNumber of iterations: {info[5]}\nMutation rate: {info[6]}\nCrossover rate: {info[7]}'
    if type==2:
        text = f'Dissagregation: {info[0]}\nFitness of last max (solution): {round(info[1], 10)}\nModularity: {info[3]}\nUsed systems: {info[2]}\n'\
        f'Population size: {info[4]} \nNumber of iterations: {info[5]}\nTournament size: {info[6]}\nMutation rate: {info[7]}\nCrossover rate: {info[8]}\nElite size: {info[9]}'

    fig, ax = plt.subplots(layout='constrained')
    iterations = [j for j in range(1, num_of_iterations+1)]
    ax.plot(iterations, avgs, label='Average')
    ax.plot(iterations, maxis, label='Maximum')
    ax.plot(iterations, minis, label='Minimum')
    ax.tick_params(axis='x', rotation=70)
    plt.xlabel('Number of iteration')
    plt.ylabel('Fitness function')
    plt.legend(loc='center right')

    fig.text(0.55, 0.15, text, size=8,
            ha="left", va="bottom",
            bbox=dict(boxstyle="square",
                    ec=(1., 0.5, 0.5),
                    fc=(1., 0.8, 0.8),
                    ),weight='bold'
            )

    plt.grid(color = 'RoyalBlue', linestyle = '--', linewidth = 0.15)
    plt.title('ID: '+unique, weight='bold', color='RoyalBlue')

    plt.show()

def make_histogram(fitness_list, unique, gen):
    fig, ax = plt.subplots(layout='constrained')
    bins = math.ceil(math.sqrt(len(fitness_list)))
    plt.hist(fitness_list, bins, histtype='step', fill=True, color='skyblue')
    ax.tick_params(axis='x', rotation=70)
    plt.xlabel('Fitness function')
    plt.ylabel('Frequency')

    plt.title('ID: '+unique, weight='bold', color='RoyalBlue')

    text = f'Generation: {gen}'

    fig.text(0.90, 0.90, text, size=15,
            ha="right", va="top",
            bbox=dict(boxstyle="square",
                    ec=(1., 0.5, 0.5),
                    fc=(1., 0.8, 0.8),
                    ),weight='bold'
            )
    
    plt.show()

def visualizeMap(links, nodes, demands_paths, pop, unique, type, info):
    fig, ax = plt.subplots(layout="constrained")
    ax.set_xlim(13.5, 25)
    ax.set_ylim(49, 55)

    edges_dict = defaultdict(lambda:0)
    
    ind = pop[-1]

    for link, paths_dict in ind.items():
        for path, edge_value  in paths_dict.items():
            demand = 'Demand_'+link.partition('_')[-1]
            for direct_path in demands_paths[demand][path]:
                edges_dict[direct_path]+=edge_value

    links_with_coordinates = defaultdict(list)

    for link, cities in links.items():
        for city in cities:
            links_with_coordinates[link].append(nodes[city])
    
    for city, v in nodes.items():
        x, y = v[0], v[1]
        ax.scatter(x,  y, color='red', s=500, marker='*', zorder=3)
        ax.annotate(f'{city}', (x+0.25, y+0.25),
                    ha='center', va='center', fontsize=14,
                    color='darkblue', zorder=4, weight='bold')
    
    weights = list(edges_dict.values())
    print(len(weights))
    for idx, (k, v) in enumerate(links_with_coordinates.items()):
        x, y = [], []
        for source, target in v:
            x.append(source)
            y.append(target)
        ax.plot(x, y, color='black', zorder=1)
        ax.annotate(int(weights[idx]), (x[0]+(x[1]-x[0])/2+0.1, y[0]+(y[1]-y[0])/2+0.1),
                    ha='center', va='center', fontsize=14,
                    color='green', zorder=2, weight='bold')
    
    fig.text(0.95, 0.95, f'Systems used: {int(info[2])}', size=20,
        ha="right", va="top",
        bbox=dict(boxstyle="square",
                ec=(1., 0.5, 0.5),
                fc=(1., 0.8, 0.8),
                ),weight='bold'
        )

    for xmaj in ax.xaxis.get_majorticklocs():
        ax.axvline(x=xmaj, ls='-', linewidth=0.5)
    for xmin in ax.xaxis.get_minorticklocs():
        ax.axvline(x=xmin, ls='-', linewidth=0.5)

    for ymaj in ax.yaxis.get_majorticklocs():
        ax.axhline(y=ymaj, ls='-', linewidth=0.5)
    for ymin in ax.yaxis.get_minorticklocs():
        ax.axhline(y=ymin, ls='-', linewidth=0.5)

    plt.title('ID: '+unique, weight='bold', color='RoyalBlue')

    res = fig.get_size_inches()

    fig.set_size_inches(res*2)
    plt.show()