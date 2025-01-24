# Data for MultiHop_GRAND with k_hops [1, 2, 4, 5, 6, 7, 8, 9, 10]
grand_next_data = [
    {"k_hops": 1, "mean": 0.4075, "std": 0.0040},
    {"k_hops": 2, "mean": 0.4046, "std": 0.0076},
    {"k_hops": 4, "mean": 0.4059, "std": 0.0068},
    {"k_hops": 5, "mean": 0.4026, "std": 0.0012},
    {"k_hops": 6, "mean": 0.4053, "std": 0.0012},
    {"k_hops": 7, "mean": 0.4080, "std": 0.0035},
    {"k_hops": 8, "mean": 0.4054, "std": 0.0052},
    {"k_hops": 9, "mean": 0.3937, "std": 0.0058},
    {"k_hops": 10, "mean": 0.3988, "std": 0.0051},
]

# Generate .txt files for MultiHop_GRAND with the next data
grand_next_file_paths = []
for entry in grand_next_data:
    file_name = f"MultiHop_GRAND_K_hops_{entry['k_hops']}.txt"
    content = f"mean: {entry['mean']}, std: {entry['std']}"
    file_path = f"{file_name}"
    grand_next_file_paths.append(file_path)
    with open(file_path, "w") as file:
        file.write(content)

grand_next_file_paths
