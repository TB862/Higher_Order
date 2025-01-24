# Data for MultiHop_GRAND with k_hops [1, 2, 9, 10]
grand_data = [
    {"k_hops": 1, "mean": 0.6824, "std": 0.0192},
    {"k_hops": 2, "mean": 0.7961, "std": 0.0200},
    {"k_hops": 9, "mean": 0.8000, "std": 0.0192},
    {"k_hops": 10, "mean": 0.8118, "std": 0.0096},
]

# Generate .txt files for MultiHop_GRAND
grand_file_paths = []
for entry in grand_data:
    file_name = f"MultiHop_GRAND_K_hops_{entry['k_hops']}.txt"
    content = f"mean: {entry['mean']}, std: {entry['std']}"
    file_path = f"{file_name}"
    grand_file_paths.append(file_path)
    with open(file_path, "w") as file:
        file.write(content)

grand_file_paths
