# Data for HoGA-GAT model
hoga_gat_data = [
    {"k_hops": 1, "mean": 0.6054, "std": 0.0530},
    {"k_hops": 2, "mean": 0.6486, "std": 0.0342},
]

# Generate .txt files for HoGA-GAT
hoga_gat_file_paths = []
for entry in hoga_gat_data:
    file_name = f"MultiHop_GAT_K_hops_{entry['k_hops']}.txt"
    content = f"mean: {entry['mean']}, std: {entry['std']}"
    file_path = f"{file_name}"
    hoga_gat_file_paths.append(file_path)
    with open(file_path, "w") as file:
        file.write(content)

hoga_gat_file_paths
