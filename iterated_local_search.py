import pandas as pd
import numpy as np
import os

def ils_algorithm(file_path, budget, output_file):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Identify the columns for configurations and performance
    config_columns = list(data.columns[:-1])
    performance_column = data.columns[-1]

    # create dictionary for quick lookup of performance based on configuration
    lookup = dict(zip(
        map(tuple, data[config_columns].values),
        data[performance_column].values
    ))
    search_space = {col: data[col].unique() for col in config_columns}

    # Optimisation logic
    system_name = os.path.basename(file_path).split(',')[0]
    if system_name.lower() == "---":
        maximisation = True
    else:
        maximisation = False
    
    # Extract the best and worst performance values
    if maximisation:
        worst_value = min(lookup.values()) / 2
        best_performance = -np.inf
    else:
        worst_value = max(lookup.values()) * 2
        best_performance = np.inf
    
    best_solution = None
    search_results = []
    valid_measurements = 0
    
    # keep track of good solutions to restart from
    elite_solutions = []
    # Track recent solutions to encourage diversity
    recent_solutions = set()

    # Parameter settings
    max_elite_size = 6  
    perturbation_prob = 0.65 
    max_stuck_count = 10  
    perturbation_magnitude = 3  

    # Main search loop
    while valid_measurements < budget:
        # restart/ perturbation step
        if elite_solutions and np.random.random() < perturbation_prob:
            # Perturb an elite solution (change more settings for better exploration)
            elite = elite_solutions[np.random.randint(0, len(elite_solutions))]
            current_sol = list(elite['sol'])
            num_changes = min(perturbation_magnitude, len(config_columns))
            indices = np.random.choice(len(config_columns), size=num_changes, replace=False)
            for idx in indices:
                current_sol[idx] = np.random.choice(search_space[config_columns[idx]])
            current_sol = tuple(current_sol)
        else:
            # Random valid start from the dataset to ensure a good initialisation
            row = data.sample(n=1).iloc[0]
            current_sol = tuple(row[config_columns])

        # Every time we 'measure' a configuration found in the lookup, it counts
        if current_sol in lookup:
            current_perf = lookup[current_sol]
            valid_measurements += 1
            search_results.append(list(current_sol) + [current_perf])
            recent_solutions.add(current_sol)
        else:
            # Invalid configurations do not consume budget
            continue 

        # Local search (Hill climbing)
        stuck_count = 0
        improvements_in_window = 0
        window_size = 5
        
        while valid_measurements < budget and stuck_count < max_stuck_count:
            # Generate a neighbour (change 1 random setting)
            neighbour = list(current_sol)
            idx_to_change = np.random.randint(0, len(config_columns))
            neighbour[idx_to_change] = np.random.choice(search_space[config_columns[idx_to_change]])
            neighbour = tuple(neighbour)
            
            if neighbour in lookup:
                neighbour_perf = lookup[neighbour]
                valid_measurements += 1 # Valid configuration: consume budget
                search_results.append(list(neighbour) + [neighbour_perf])
                
                # Check for improvement
                improved = neighbour_perf > current_perf if maximisation else neighbour_perf < current_perf
                if improved:
                    current_sol, current_perf = neighbour, neighbour_perf
                    stuck_count = 0 # Reset because a better neighbour is found
                    improvements_in_window += 1
                else:
                    stuck_count += 1
            else:
                # Invalid neighbour checks do not consume budget 
                # Increment stuck_count to avoid getting trapped in an empty area
                stuck_count += 1
        
        # Update Global Best and Elites
        is_global_best = current_perf > best_performance if maximisation else current_perf < best_performance
        if is_global_best:
            best_performance = current_perf
            best_solution = current_sol
            elite_solutions.append({'sol': current_sol, 'perf': current_perf})
            # Keep elite list small and sorted to maintain quality with more diversity
            elite_solutions = sorted(elite_solutions, key=lambda x: x['perf'], reverse=maximisation)[:max_elite_size]
            
    # 4. Save results to CSV
    columns = list(config_columns) + ["Performance"]
    pd.DataFrame(search_results, columns=columns).head(budget).to_csv(output_file, index=False)
    return [int(x) for x in best_solution], best_performance

def main():
    datasets_folder = "datasets"
    output_folder = "pils_results"
    os.makedirs(output_folder, exist_ok=True)
    budget = 100

    results = {}
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(datasets_folder, file_name)
            output_file = os.path.join(output_folder, f"{file_name.split('.')[0]}_pils_results.csv")
            best_solution, best_performance = ils_algorithm(file_path, budget, output_file)
            results[file_name] = {
                "Best Solution": best_solution,
                "Best Performance": best_performance
            }

    # Print the results
    for system, result in results.items():
        print(f"System: {system}")
        print(f"  Best Solution:    [{', '.join(map(str, result['Best Solution']))}]")
        print(f"  Best Performance: {result['Best Performance']}")

if __name__ == "__main__":
    main()