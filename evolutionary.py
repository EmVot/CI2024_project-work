from treeMap import generate_random_expression as spawn
from treeMap import treeMap
import numpy as np
from copy import deepcopy
import random
import pdb
from constantsGenerator import coefficient_range

def spawn_offspring(individuals:int, max_depth:int, constant_range:tuple, problem:np.ndarray ):
    '''
    Generates an offspring of 'individuals' number, which can reach a depth of 'max_depth'
    An individual is a treeMap referring to a numpy expression valid for the specified problem
    Params:
    individuals (int): number of individuals
    max_depth (int): max depth of the expression generated
    x_vals (np.ndarray): the problem x values
    Returns:
    The offspring
    '''

    x_values=problem['x']

    variables_coefficients_dict=coefficient_range(problem)
    offspring=[]
    constants=np.linspace(constant_range[0],constant_range[1],50) + 1e-9 #avoid generating null costants
    constants = constants.tolist()
    variables = [f"x{index}" for index in range(x_values.shape[0])]
    
    for _ in range(individuals):

        variables_dict=dict(zip(variables,[0] * len(variables)))
        expr = spawn(max_depth,deepcopy(variables_dict),constants,variables_coefficients_dict)
        y_values = [expr.validate_and_evaluate(dict(zip(variables,x))) for x in x_values.T]

        while False in y_values:
            expr = spawn(max_depth,deepcopy(variables_dict),constants,variables_coefficients_dict)
            y_values = [expr.validate_and_evaluate(dict(zip(variables,x))) for x in x_values.T]
        
        newChild = treeMap(max_depth,variables,constants,expr)
        offspring.append(newChild)

    return offspring


def crossover(parents: list, x_values: np.ndarray):
    """
    Performs subtree crossover ensuring the resulting offspring do not exceed allowed_depth.
    """
    
    allowed_depth = parents[0].max_depth+1

    def get_random_node(root, max_depth, current_depth=0):
        """
        Select a random node avoiding those at max depth.
        """
        if root is None or current_depth >= max_depth - 1:
            return root  # Return current node if too deep

        candidates = [root]  # Candidates list
        if root.left_child:
            candidates.append(get_random_node(root.left_child, max_depth, current_depth + 1))
        if root.right_child:
            candidates.append(get_random_node(root.right_child, max_depth, current_depth + 1))

        return random.choice(candidates)  # Return a random node among visited ones

    def subtree_crossover(tree1, tree2):
        """
        Performs subtree crossover between two trees with depth control.
        """
        max_attempts = 10  # Prevent infinite loops
        for _ in range(max_attempts):
            # Clone the trees
            tree1_copy = deepcopy(tree1)
            tree2_copy = deepcopy(tree2)

            # Select two random nodes
            node1 = get_random_node(tree1_copy.root, tree1_copy.getDepth())
            node2 = get_random_node(tree2_copy.root, tree2_copy.getDepth())

            # Swap the subtrees
            node1.value, node2.value = node2.value, node1.value
            node1.left_child, node2.left_child = node2.left_child, node1.left_child
            node1.right_child, node2.right_child = node2.right_child, node1.right_child

            offspring=[]
            # Check if the new trees exceed allowed depth
            if tree1_copy.getDepth() <= allowed_depth:
                offspring.append(tree1_copy)
            if tree2_copy.getDepth() <= allowed_depth:
                offspring.append(tree2_copy)
            
            return offspring


    # Select parents randomly
    parent1, parent2 = random.sample(parents, 2)

    # Perform crossover with depth control
    offspring = subtree_crossover(parent1, parent2)

    # Validate offspring before returning
    offspring = [child for child in offspring if child.validate_and_evaluate(x_values)]

    return offspring


def mutation(offspring,max_depth,problem,constants) -> list:

    """
    Applies genetic mutation to a list of offspring using subtree mutation and hoist mutation.
    
    This function iterates over a population of offspring and applies one of two mutation
    strategies with predefined probabilities:
    
    1. **Subtree Mutation (70% probability)**: A randomly chosen subtree is replaced with
        a newly generated random subtree.
    2. **Hoist Mutation (30% probability)**: A randomly chosen subtree is promoted to replace
        its parent, effectively reducing tree depth and potentially mitigating bloat.
    
    If the mutated offspring passes validation (`validate_and_evaluate()`), it is added to the
    new mutated offspring population. Otherwise, the original individual is retained.

    Parameters:
        offspring (list): A list of `treeMap` objects representing the current population.
        max_depth (int): The maximum depth allowed for generated subtrees in mutation.
        x_values (list): The list of available variables in the expressions.
        constants (list): A list of numerical constants available for the expressions.

    Returns:
        list: A new list of `treeMap` objects containing the mutated offspring.
    """
    

    x_values=problem['x']

    variables_coefficients_dict=coefficient_range(problem)
    
    mutated_offspring = []

    def subtree_mutation(tree, max_depth, variables_dict, constants) -> treeMap:
        
        """
        Perform subtree mutation on a tree by replacing a randomly chosen subtree
        with a newly generated random subtree.
        
        Params:
            tree: The treeMap object.
            max_depth: Maximum depth for the considered problem (avoid boalting).
            variables_dict: Dictionary of variavles available x nur of times that variable hs been picked
            constants: List of available constants.
            
        Returns:
            A new treeMap with the mutated subtree.
        """

        tree_copy = deepcopy(tree)
        
        def mutate_node(node,current_depth):
            if node is None:
                return None
            
            if node.value in variables_dict:
                variables_dict[node.value]+=1

            if random.random() < 0.1 and current_depth < max_depth:  # Probability of mutation
                if node.value in variables_dict or isinstance(node.value,(float,int,np.float64)):
                    #if the node is a leaf we do not want the expression to grow too deep
                    if node.value in variables_dict:
                        variables_dict[node.value]-=1
                    #therefore the expression maximum depth remains unchanged
                    return spawn(0, variables_dict, constants,variables_coefficients_dict)
                else:
                    return spawn(random.randint(1, max_depth-current_depth+1), variables_dict, constants,variables_coefficients_dict)
            
            else:
                if node.left_child:
                    node.left_child = mutate_node(node.left_child,current_depth+1)
                if node.right_child:
                    node.right_child = mutate_node(node.right_child,current_depth+1)
            return node
        
        tree_copy.root = mutate_node(tree_copy.root,0)

        return tree_copy

    def hoist_mutation(tree) -> treeMap:
        """
        Perform hoist mutation by selecting a random subtree and replacing
        the original tree with a subtree of it.
        
        Params:
            tree: The treeMap object.
            
        Returns:
            A new treeMap with the hoist mutation applied.
        """

        tree_copy = deepcopy(tree)
        
        def select_random_subtree(node):
            """
            Recursively selects a random subtree.
            """
            if node is None or (node.left_child is None and node.right_child is None):
                return node  # Return leaf node
            if random.random() < 0.3:  # Probability of selecting current node
                return node
            elif node.left_child and random.random() < 0.5:
                return select_random_subtree(node.left_child)
            elif node.right_child:
                return select_random_subtree(node.right_child)
            return node

        selected_subtree = select_random_subtree(tree_copy.root)
        
        # Replace the original tree with the selected subtree
        if selected_subtree:
            tree_copy.root = deepcopy(selected_subtree)

        return tree_copy

    variables = [f"x{index}" for index in range(x_values.shape[0])]
    variables_dict=dict(zip(variables,[0] * len(variables)))

    #we need to keep under control the tree bloating -> dynamic probability of undergoing a hoist mutation

    for individual in offspring:
            
            #individual_depth=individual.getDepth()
            #p=1-(max_depth/individual_depth)

        if random.random() < 0.7: #we prefer adopting a subtree_mutation

            mutated_child=subtree_mutation(individual,max_depth,variables_dict,constants)
        else:
            mutated_child=hoist_mutation(individual)
        
        if mutated_child.validate_and_evaluate(x_values):
            mutated_offspring.append(mutated_child)
        else:
            mutated_offspring.append(individual)
    
    return mutated_offspring

# evolutionary cycle:
# Population of size n
# 1) create offspring
#   -> perform crossover and obtain an offspring population of size o
#   -> perform mutation on the offspring
# 2) tournament selection
#   -> put together parents and children, obtaining a population size of n + o
#   -> perform tournament selection and obtain a population of size n
# 3) repeat

def create_offspring(parents,offspringSize,problem,constants,max_depth)->list[treeMap]:
    
    offspring=[]
    x_values=problem['x']

    while len(offspring) < offspringSize:
        offspring.extend(crossover(parents,x_values))

    
    offspring=mutation(offspring,max_depth,problem,constants)

    return offspring

def tournament_selection(population:list,selective_pressure:int,problem:np.ndarray)->list[treeMap]:

    survivors = []

    while len(population)>0:

        #select the partecipants and remove them from the population
        arena=[]

        for _ in range(selective_pressure):

            if len(population)<=0:
                break
            arena.append(population.pop(random.choice(range(len(population)))))

        #sort them based on their fitness
        fitnesses=[ind.fitness(problem) for ind in arena]
        arena=np.array(arena)[np.argsort(fitnesses)].tolist()
        #only half of the arena survives the selection
        survivors.extend(arena[:len(arena)//2])

    return survivors

def evolutionary_algorithm(population_size,offspring_size,generations,selective_presure,max_expression_depth,problem,const_range)->treeMap:

    constant_range=np.linspace(const_range[0],const_range[1],10)

    population=spawn_offspring(population_size,max_expression_depth,constant_range,problem)

    for gen in range(generations):

        print(f'Geneartion: {gen}')
        #1: create offspring
        offspring=create_offspring(population,offspring_size,problem,constant_range,max_expression_depth)
        #2: tournament selection and parent selection (they are made both in tournament slection)
        population.extend(offspring)
        population=tournament_selection(population,selective_presure,problem)
        

    fitnesses=list(map(lambda ind: ind.fitness(problem),population))
    
    return np.array(population)[np.argsort(fitnesses)][0]