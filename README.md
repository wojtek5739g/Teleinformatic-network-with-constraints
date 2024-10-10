# Project overview

# Telecommunication Network Design Using Evolutionary Algorithm

## Problem Statement
The objective of this project is to design a telecommunication network that minimizes the number of transmission systems with different modularity levels \( m \) (where \( m = 1 \), \( m > 1 \), and \( m >> 1 \)) using an evolutionary algorithm. The network is represented as a graph \(G = (N, E)\), where:
- \(N\) is the set of nodes
- \(E\) is the set of edges

The capacity function for each edge is given by:

![image](https://github.com/user-attachments/assets/f8b785c1-3cc4-46fb-a780-663d87e5bfab)


where \(o\) is the demand and \(m\) is the modularity of the transmission system. The set of demands \(D\) between each pair of nodes is described by a demand matrix. For each demand, at least two predefined paths are available for transmission.

## Objective
This project aims to evaluate how demand aggregation affects the cost of the network in terms of the number of transmission systems used. The two cases to be considered are:
1. **Aggregation**: The demand is routed through a single path.
2. **Disaggregation**: The demand is distributed across all available paths.

Additionally, we aim to optimize the following evolutionary algorithm parameters:
- Probabilities of genetic operators (crossover and mutation)
- Population size

## Evolutionary Algorithm Approach
### Key Steps:
1. **Initial Population**: Generate an initial population of potential network configurations.
2. **Fitness Evaluation**: Evaluate each solution's fitness based on the number of transmission systems required.
3. **Selection**: Select individuals from the population for reproduction based on their fitness.
4. **Crossover & Mutation**: Apply crossover and mutation operators to produce new offspring.
5. **Replacement**: Replace the old population with a new generation based on fitness.
6. **Termination**: The algorithm terminates when a specified number of generations is reached or when convergence occurs.

### Parameters to Optimize:
- **Crossover Probability**: Determines the frequency of crossover between individuals.
- **Mutation Probability**: Controls the likelihood of mutations to introduce variability.
- **Population Size**: The number of individuals in each generation.

## Data Source
The network topology and demand matrix for this project are taken from the **SNDlib** dataset for the Polish network. The data can be downloaded from the following link:

[SNDlib Polish Network Data](http://sndlib.zib.de/home.action)

## Experimental Setup
1. **Aggregation vs. Disaggregation**: Comparing the cost in terms of the number of transmission systems used when all demand is routed through a single path (aggregation) versus when demand is distributed across multiple paths (disaggregation).
2. **Genetic Operator Optimization**: Conducting experiments to find the optimal values for crossover and mutation probabilities, as well as the ideal population size.

# Usage

In order to launch simulation type for example: 

```python
python EvolutionAlgorithm.py --d 1 --m 1 --t 1 --mid 5 --p 150 --n 10 --ts 2 --mr 0.01 --cr 0.7 --es 10
```

# Other

To check descriptions of the perticular parameters type: python EvolutionAlgorithm.py --help

To see detailed documentation: go to POP___Sieć_teleinformatyczna_z_ograniczeniami.pdf (ENGLISH VERSION TO BE ADDED)

Example solution map (taken from the documentation):
![image](https://github.com/user-attachments/assets/1d177d89-eee6-488f-a670-159f5afc7660)


