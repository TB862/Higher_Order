# Data for MultiHop_GAT with k_hops from 1 to 10
gat_shops_data = [
    {"k_hops": 1, "mean": 0.771, "std": 0.0403},
    {"k_hops": 2, "mean": 0.748, "std": 0.0225},
    {"k_hops": 3, "mean": 0.7504, "std": 0.0174},
    {"k_hops": 4, "mean": 0.7416, "std": 0.0196},
    {"k_hops": 5, "mean": 0.7268, "std": 0.0205},
    {"k_hops": 6, "mean": 0.7354, "std": 0.0208},
    {"k_hops": 7, "mean": 0.7402, "std": 0.0126},
    {"k_hops": 8, "mean": 0.7248, "std": 0.0382},
    {"k_hops": 9, "mean": 0.7026, "std": 0.0131},
    {"k_hops": 10, "mean": 0.7056, "std": 0.0319},
]

# Generate .txt files for MultiHop_GAT with shops data
gat_shops_file_paths = []
for entry in gat_shops_data:
    file_name = f"MultiHop_GAT_K_hops_{entry['k_hops']}.txt"
    content = f"mean: {entry['mean']}, std: {entry['std']}"
    file_path = f"{file_name}"
    gat_shops_file_paths.append(file_path)
    with open(file_path, "w") as file:
        file.write(content)

gat_shops_file_paths
