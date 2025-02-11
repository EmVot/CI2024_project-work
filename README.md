# CI2024_project-work
SymReg project for Computational Intelligence Course


## Overview

This project implements an **Evolutionary Algorithm (EA)** for **Symbolic Regression**, which aims to find mathematical expressions that best fit a given dataset. The core idea is to evolve populations of expression trees over multiple generations using genetic programming techniques.
For more insights please read the `Project work` section of my report


## Code Organization
### src
#### Contains all the source material used for representation and evolutionary algorithm.
`evolutionary.py` contains all the methods used for the EA, from crossover to tournament selection.
Its main function is `evolutionary_algorithm` which acts as a general wrapper for the evolutionary process.

`node.py` contains the individual representation, which will be explained in detail in the following section.

`treeMap.py` contains the main wrapper class for the expression representation, with the omonimus class.
Moreover the function `generate_random_expression` represent the main function for the generation of valid random expressions (with respect to a given problem)

`utils.py` contains the functions for expression representation.

`constantsGenerator.py` contains functions for variable coefficient generator, with the main function `coefficient_range` (see report and code docstring for more details).

`train.py` acts as a train wrapper function for alle the problems given. It defines all the variables used for EA.

`operatos.py` is a collection of operators considered.

### data folder
#### Contains all the data used for training


## Individual Representation

The candidate solutions are represented as **tree structures**, in `node.py` where:

- **Leaf nodes** are either constants or variables (e.g., `x1`, `x2`).
- **Internal nodes** are operators (`+`, `-`, `*`, `/`, etc.) or unary functions (`sin`, `cos`, `exp`, etc.).
- Each tree corresponds to a mathematical expression.

The tree structure ensures that all candidate solutions remain valid mathematical expressions.
---

### Population collection

A population consists of an array of TreeMap objects, which are a wrapper representation of the tree cited above and include class methods to copute fitness, evaluate and validate the expression they embed and a method to visualize a readible expression. Implementation can be found in `treeMap.py`

### Evolutionary algorithm

The python file `evolutionary.py` contains all the specs about the EA.
- `spawn offspring` acts as a wrapper function for the popoulation initialization for the current problem
- `create_offspring` performs crossover and mutation to create offspring till the maximum number of individual is reached
- `tournament_selection` performs tournament selction for a specified selective pressure (default: 8)

### Additional utilities
- Contained in `constatnsGenerator.py` we find a generator for variable coefficients scaled with respect to the problem overall output magnitude
- `utils.py` keeps the functions for the tree visualization


## Evolutionary Algorithm Overview

The evolutionary algorithm follows these steps for `N` generations:

1. **Crossover:** Swap subtrees between parents to create new offspring.
2. **Mutation:** Modify subtrees to introduce diversity.
3. **Survivor Selection:** Select the best individuals to continue to the next generation.
4. **Repeat** until the termination criterion is met.

---

## Initialization

- The function `spawn_offspring()` creates an initial population of random expression trees.
- Variables and constants are selected probabilistically to maintain diversity.
- The generated expressions are validated to avoid domain errors (e.g., `sqrt` of negative numbers).

---

## Crossover and Mutation

### **Crossover**

- Subtree crossover swaps randomly selected subtrees between two parents.
- Ensures that offspring do not exceed the allowed depth.
- Uses `subtree_crossover()` to generate valid offspring.

### **Mutation**

- Two types of mutation are applied:
  1. **Subtree Mutation:** Replaces a random subtree with a newly generated one.
  2. **Hoist Mutation:** Promotes a randomly selected subtree to replace its parent.
- Mutation helps maintain diversity and prevents premature convergence.

---

## Fitness Computation

- The fitness function is based on **Mean Squared Error (MSE)** between the predicted and actual outputs:

  \(MSE = \frac{1}{n} \sum (y_{true} - y_{pred})^2\)

- Implemented in `treeMap.fitness()`.

- Lower MSE values indicate better-fitting expressions.

---

## Running the Algorithm

The main evolutionary loop is implemented in `evolutionary_algorithm()`:

```python
def evolutionary_algorithm(population_size, offspring_size, generations, selective_pressure, max_expression_depth, problem, const_range):
    population = spawn_offspring(population_size, max_expression_depth, const_range, problem)
    for gen in range(generations):
        offspring = create_offspring(population, offspring_size, problem, const_range, max_expression_depth)
        population.extend(offspring)
        population = tournament_selection(population, selective_pressure, problem)
    return min(population, key=lambda ind: ind.fitness(problem))
```

This function runs the evolutionary process and returns the best solution found.

---

## Conclusion

This evolutionary approach enables the discovery of mathematical expressions that best approximate given data. By leveraging symbolic regression and genetic programming, it finds interpretable solutions without requiring predefined model structures.
All the resuld discussion can be found in the report final part.

