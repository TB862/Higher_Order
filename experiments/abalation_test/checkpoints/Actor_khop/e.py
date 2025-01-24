# Data for MultiHop_GAT with k_hops [1, 2, 6, 7, 8, 9, 10]
gat_additional_data = [
    {"k_hops": 1, "mean": 0.2826, "std": 0.0147},
    {"k_hops": 2, "mean": 0.2867, "std": 0.0152},
    {"k_hops": 6, "mean": 0.2829, "std": 0.0105},
    {"k_hops": 7, "mean": 0.2892, "std": 0.0110},
    {"k_hops": 8, "mean": 0.2921, "std": 0.0097},
    {"k_hops": 9, "mean": 0.2922, "std": 0.0065},
    {"k_hops": 10, "mean": 0.2829, "std": 0.0094},
]

# Generate .txt files for MultiHop_GAT with additional data
gat_additional_file_paths = []
for entry in gat_additional_data:
    file_name = f"MultiHop_GAT_K_hops_{entry['k_hops']}.txt"
    content = f"mean: {entry['mean']}, std: {entry['std']}"
    file_path = f"{file_name}"
    gat_additional_file_paths.append(file_path)
    with open(file_path, "w") as file:
        file.write(content)

gat_additional_file_paths
