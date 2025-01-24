# Data for MultiHop_GAT model over k-hops [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
gat_data = [
    {"k_hops": 1, "mean": 0.8720, "std": 0.0241},
    {"k_hops": 2, "mean": 0.8826, "std": 0.0054},
    {"k_hops": 3, "mean": 0.8766, "std": 0.0128},
    {"k_hops": 4, "mean": 0.8742, "std": 0.0115},
    {"k_hops": 5, "mean": 0.8772, "std": 0.0191},
    {"k_hops": 6, "mean": 0.8500, "std": 0.0241},
    {"k_hops": 7, "mean": 0.8634, "std": 0.0191},
    {"k_hops": 8, "mean": 0.8552, "std": 0.0177},
    {"k_hops": 9, "mean": 0.8492, "std": 0.0143},
    {"k_hops": 10, "mean": 0.8623, "std": 0.0134},
]

# Generate .txt files for MultiHop_GAT
gat_file_paths = []
for entry in gat_data:
    file_name = f"MultiHop_GAT_K_hops_{entry['k_hops']}.txt"
    content = f"mean: {entry['mean']}, std: {entry['std']}"
    file_path = f"{file_name}"
    gat_file_paths.append(file_path)
    with open(file_path, "w") as file:
        file.write(content)

gat_file_paths
