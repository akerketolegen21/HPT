import json
import os
from collections import defaultdict
import argparse
import torch

def convert_data_format(input_path, output_path, label_to_id=None):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    if not label_to_id:
        # Collect all unique labels
        all_labels = set()
        for item in data:
            all_labels.update(item["labels"])
        
        label_to_id = {label: idx for idx, label in enumerate(sorted(all_labels))}
        
        # Save label mapping
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(os.path.join(os.path.dirname(output_path), "label_map.json"), 'w') as f:
            json.dump(label_to_id, f, indent=2)
    
    formatted_data = []
    for item in data:
        label_ids = [label_to_id[label] for label in item["labels"]]
        
        formatted_item = {
            "id": item["id"],
            "text": item["text"],
            "labels": label_ids,
            "raw_labels": item["labels"]  # Keep original labels for reference
        }
        formatted_data.append(formatted_item)
    
    # Save converted data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(formatted_data, f, indent=2)
    
    return label_to_id

def process_hierarchy_tree(output_path, label_to_id):
    """
    Process hierarchy tree and convert to HPT format.
    """
    from hierarchical_structure import T as hierarchy_data 

    # Convert to the format HPT expects: {parent_id: [child_id1, child_id2, ...]}
    hierarchy_dict = defaultdict(set)
    for node in hierarchy_data.nodes():
        if node not in label_to_id and node != "root" and node != "propagandistic" and node != "non-propagandistic":
            # Assign a new ID for nodes not in label_to_id
            next_id = max(label_to_id.values()) + 1
            label_to_id[node] = next_id
            print(f"Added node {node} with ID {next_id} to label_to_id")
    
    # Now construct the hierarchy dictionary
    for node in hierarchy_data.nodes():
        if node in label_to_id:  # Only process nodes with IDs
            node_id = label_to_id[node]
            children = list(hierarchy_data.successors(node))
            if children:
                # Convert child nodes to IDs
                child_ids = []
                for child in children:
                    if child in label_to_id:
                        child_ids.append(label_to_id[child])
                
                if child_ids:  # Only include if there are valid child IDs
                    hierarchy_dict[node_id] = child_ids
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(hierarchy_dict, f, indent=2)
    
    print(f"Hierarchy processed and saved to {output_path}")
    return hierarchy_dict, label_to_id

def create_value_dict_and_slot(output_dir, label_to_id, hierarchy_dict=None):
    label_dict = {v: k for k, v in label_to_id.items()}
    torch.save(label_dict, os.path.join(output_dir, "value_dict.pt"))

    hierarchy = defaultdict(set)
    for i in hierarchy_dict[0]:
        for j in hierarchy_dict[0][i]:
            hierarchy[i].add(j)
    
    torch.save(hierarchy, os.path.join(output_dir, "slot.pt"))

    
def generate_config_file(dataset_name, output_path, num_labels, hierarchy_depth):
    """
    Generate YAML configuration file for HPT.
    """
    config = f"""dataset:
  name: {dataset_name}
  task: multi-label
  max_length: 512
  level: {hierarchy_depth}
  num_labels: {num_labels}
  label_names: []
  test_mode: False
  
model:
  name: bert
  backbone: bert-base-uncased
  prompt_length: 5
  prompt_mid_dim: 512
  hyper_dims: [768, 512]
  
train:
  batch_size: 8
  optimizer: AdamW
  learning_rate: 3e-5
  weight_decay: 0.01
  gradient_accumulation_steps: 1
  max_epochs: 20
  lr_scheduler: linear
  warmup_ratio: 0.1
  eval_patience: 5
  eval_every: 100
"""
    
    with open(output_path, 'w') as f:
        f.write(config)

def main():
    parser = argparse.ArgumentParser(description="Convert data to HPT format")
    parser.add_argument("--train", required=True, help="Path to train.json")
    parser.add_argument("--val", required=True, help="Path to val.json")
    parser.add_argument("--test", required=True, help="Path to test.json")
    parser.add_argument("--output_dir", default="data/your_dataset", help="Output directory")
    parser.add_argument("--dataset_name", default="your_dataset", help="Dataset name")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Processing train data...")
    label_to_id = convert_data_format(
        args.train, 
        os.path.join(args.output_dir, "train.json")
    )
    
    print("Processing validation data...")
    convert_data_format(
        args.val, 
        os.path.join(args.output_dir, "dev.json"),  # HPT uses "dev" instead of "val"
        label_to_id
    )
    print("Processing test data...")
    
    convert_data_format(
        args.test, 
        os.path.join(args.output_dir, "test.json"),
        label_to_id
    )
    # Process hierarchy tree
    print("Processing hierarchy tree...")
    hierarchy_dict = process_hierarchy_tree(
        os.path.join(args.output_dir, "hierarchy.json"),
        label_to_id
    )
    create_value_dict_and_slot(args.output_dir, label_to_id, hierarchy_dict)
    
    # # Generate config file
    print("Generating config file...")
    generate_config_file(
        args.dataset_name,
        f"config.yaml",
        len(label_to_id),
        hierarchy_depth=3  # Update this based on your hierarchy
    )
    
    print(f"Data processing complete. Files saved to {args.output_dir}")

if __name__ == "__main__":
    main()