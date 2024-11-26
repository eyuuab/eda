def generate_truth_combinations(variables, target_values=(True, False)):
    """
    Recursively generate all combinations of truth values for the given variables
    and target values.

    Args:
        variables (tuple): A tuple of variable names (e.g., ('A', 'B', 'C')).
        target_values (tuple): A tuple of truth values to use (default: (True, False)).

    Returns:
        list of tuples: Each tuple contains the target value followed by (variable, value) pairs.
    """
    def helper(index):
        # Base case: If all variables are processed, return an empty combination
        if index == len(variables):
            return [[]]

        # Recursive case: Get combinations for the remaining variables
        combinations = helper(index + 1)
        current_var = variables[index]
        result = []

        for truth_value in target_values:
            for combination in combinations:
                # Add the current variable and its value to each combination
                result.append([(current_var, truth_value)] + combination)

        return result

    # Generate combinations and attach the target truth values
    all_combinations = []
    for target in target_values:
        for combination in helper(0):
            all_combinations.append((target, *combination))
    
    return all_combinations

# Example usage:
variables = ('A', 'B', 'C')
combinations = generate_truth_combinations(variables)
for combo in combinations:
    print(combo)
