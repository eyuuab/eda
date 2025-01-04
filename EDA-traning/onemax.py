import random

'''
    oneMax problem using Estimation of Distribution Algorithm (EDA)
'''
# initializing Problem Parameters
n = 10  
population_size = 10 
generations = 30  
top_k = 5  

# Fitness Function: Count number of 1s in the binary string
def fitness(individual):
    return sum(individual)

# Generate an individual (random binary string)
def random_individual(n):
    return [random.randint(0, 1) for _ in range(n)]

# Generate initial population
def initialize_population(pop_size, n):
    return [random_individual(n) for _ in range(pop_size)]

# Select the top k individuals based on fitness
def select_top_individuals(population, k):
    return sorted(population, key=fitness, reverse=True)[:k]


# Learn probability distribution from selected individuals
def learn_distribution(top_individuals, n):
    probabilities = [0.0] * n
    for i in range(n):
        ones_count = sum(ind[i] for ind in top_individuals)
        probabilities[i] = ones_count / len(top_individuals)
    return probabilities


# Sample new individuals from the learned distribution
def sample_new_individual(probabilities):
    
    return [1 if random.random() < p else 0 for p in probabilities]


# Main EDA Loop
def eda_one_max(n, population_size, generations, top_k):
    population = initialize_population(population_size, n)

    for generation in range(generations):
        top_individuals = select_top_individuals(population, top_k)
        probabilities = learn_distribution(top_individuals, n)
        population = [sample_new_individual(probabilities) for _ in range(population_size)]
        
        best = max(population, key=fitness)
        print(f"Generation {generation+1}: Best Fitness = {fitness(best)}, Best Individual = {best}")
        print(f"Probabilities = {probabilities}")
        print('-' * 50)
    return max(population, key=fitness)




# Run the EDA
best_solution = eda_one_max(n, population_size, generations, top_k)
print("\nFinal Best Solution:", best_solution)
print("Final Fitness:", fitness(best_solution))
















# import numpy as np
# from typing import Tuple, List
# import matplotlib.pyplot as plt

# class OneMaxUMDA:
#     def __init__(self, problem_size: int, population_size: int, selection_rate: float = 0.5):
#         """
#         Initialize the UMDA for OneMax problem.
        
#         Args:
#             problem_size: Length of binary string
#             population_size: Number of individuals in population
#             selection_rate: Proportion of population to select for model building
#         """
#         self.problem_size = problem_size
#         self.population_size = population_size
#         self.selection_size = int(population_size * selection_rate)
#         self.population = None
#         self.probabilities = np.full(problem_size, 0.5)  # Initial 50% probability
        
#     def onemax_fitness(self, individual: np.ndarray) -> int:
#         """Calculate fitness as sum of 1s in the binary string."""
#         return np.sum(individual)
    
#     def evaluate_population(self) -> np.ndarray:
#         """Evaluate fitness for entire population."""
#         return np.array([self.onemax_fitness(ind) for ind in self.population])
    
#     def select_best(self, fitness: np.ndarray) -> np.ndarray:
#         """Select best individuals based on fitness."""
#         selected_indices = np.argsort(fitness)[-self.selection_size:]
#         return self.population[selected_indices]
    
#     def update_probabilities(self, selected: np.ndarray):
#         """Update probability model based on selected individuals."""
#         self.probabilities = np.mean(selected, axis=0)
#         # Prevent convergence to absolute 0 or 1
#         self.probabilities = np.clip(self.probabilities, 0.01, 0.99)
    
#     def generate_population(self):
#         """Generate new population based on current probability model."""
#         self.population = np.random.binomial(1, self.probabilities, 
#                                            size=(self.population_size, self.problem_size))
    
#     def run(self, max_generations: int) -> Tuple[List[float], List[float]]:
#         """
#         Run the UMDA algorithm.
        
#         Args:
#             max_generations: Maximum number of generations to run
            
#         Returns:
#             Tuple of (best_fitness_history, mean_fitness_history)
#         """
#         best_fitness_history = []
#         mean_fitness_history = []
        
#         # Initialize population
#         self.generate_population()
        
#         for generation in range(max_generations):
#             # Evaluate current population
#             fitness_values = self.evaluate_population()
            
#             # Record statistics
#             best_fitness = np.max(fitness_values)
#             mean_fitness = np.mean(fitness_values)
#             best_fitness_history.append(best_fitness)
#             mean_fitness_history.append(mean_fitness)
            
#             # Check if optimal solution found
#             if best_fitness == self.problem_size:
#                 break
                
#             # Select best individuals
#             selected = self.select_best(fitness_values)
            
#             # Update probability model
#             self.update_probabilities(selected)
            
#             # Generate new population
#             self.generate_population()
        
#         return best_fitness_history, mean_fitness_history

# def plot_convergence(best_history: List[float], mean_history: List[float]):
#     """Plot convergence of the algorithm."""
#     plt.figure(figsize=(10, 6))
#     plt.plot(best_history, label='Best Fitness')
#     plt.plot(mean_history, label='Mean Fitness')
#     plt.xlabel('Generation')
#     plt.ylabel('Fitness')
#     plt.title('UMDA Convergence on OneMax Problem')
#     plt.legend()
#     plt.grid(True)
#     return plt

# # Example usage and testing
# if __name__ == "__main__":
#     # Problem parameters
#     PROBLEM_SIZE = 100
#     POPULATION_SIZE = 200
#     MAX_GENERATIONS = 50
    
#     # Initialize and run algorithm
#     umda = OneMaxUMDA(PROBLEM_SIZE, POPULATION_SIZE)
#     best_history, mean_history = umda.run(MAX_GENERATIONS)
    
#     # Print results
#     print(f"Final best fitness: {best_history[-1]}/{PROBLEM_SIZE}")
#     print(f"Number of generations: {len(best_history)}")
    
#     # Create convergence plot
#     plt = plot_convergence(best_history, mean_history)
#     plt.show()