import json
import os
from collections import defaultdict
import argparse
import torch
import re

def preprocess_text(text):
    """
    Preprocess text as described in the paper:
    - Add [CLS] tokens between sentences separated by \n\n
    """
    if not text:
        return text
    
    # Split by double newlines (different picture boxes in memes)
    sentences = text.split('\n\n')
    
    # Join with [CLS] token as mentioned in the paper
    processed_text = ' [CLS] '.join(sentence.strip() for sentence in sentences if sentence.strip())
    
    return processed_text

def expand_labels_with_hierarchy(labels, label_mapping):
    """
    Expand labels that have multiple parents in the hierarchy.
    For example, "Whataboutism" becomes both "Distraction_Whataboutism" and "AdHominem_Whataboutism"
    """
    expanded_labels = []
    
    for label in labels:
        if label in label_mapping:
            # Add all duplicated versions of this label
            expanded_labels.extend(label_mapping[label])
        else:
            # Keep original label if no mapping exists
            expanded_labels.append(label)
    
    return list(set(expanded_labels))  # Remove duplicates

def convert_data_format(input_path, output_path, label_to_id=None, label_mapping=None):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    if not label_to_id:
        # First pass: collect all unique labels including expanded ones
        all_labels = set()
        for item in data:
            if "labels" in item and item["labels"]:
                original_labels = item["labels"]
                if label_mapping:
                    expanded_labels = expand_labels_with_hierarchy(original_labels, label_mapping)
                    all_labels.update(expanded_labels)
                else:
                    all_labels.update(original_labels)
        
        # Add hierarchy nodes
        hierarchy_nodes = [
            "root", "propagandistic", "non-propagandistic", 
            "Logos", "Ethos", "Pathos", "Reasoning", "Justification", 
            "Simplification", "Distraction", "Ad Hominem"
        ]
        all_labels.update(hierarchy_nodes)
        
        label_to_id = {label: idx for idx, label in enumerate(sorted(all_labels))}
        
        # Save label mapping
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(os.path.join(os.path.dirname(output_path), "label_map.json"), 'w') as f:
            json.dump(label_to_id, f, indent=2)
    
    formatted_data = []
    for item in data:
        # Preprocess text as described in the paper
        processed_text = preprocess_text(item["text"])
        if "labels" in item and item["labels"]:
            # Expand labels with hierarchy mapping
            original_labels = item["labels"]
            if label_mapping:
                final_labels = expand_labels_with_hierarchy(original_labels, label_mapping)
            else:
                final_labels = original_labels
            
            label_ids = [label_to_id[label] for label in final_labels if label in label_to_id]
            
            formatted_item = {
                "id": item["id"],
                "text": processed_text,  # Use preprocessed text
                "labels": label_ids,
                "raw_labels": original_labels  # Keep original labels for reference
            }
        else:
            formatted_item = {
                "id": item["id"],
                "text": processed_text,
                "labels": [],  # Empty list for test data
                "raw_labels": []  # Empty list for test data
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
    hierarchy_dict = {}
    
    # Build the hierarchy dictionary with proper ID mapping
    for node in hierarchy_data.nodes():
        if node in label_to_id:
            node_id = label_to_id[node]
            children = list(hierarchy_data.successors(node))
            
            if children:
                child_ids = []
                for child in children:
                    if child in label_to_id:
                        child_ids.append(label_to_id[child])
                
                if child_ids:
                    hierarchy_dict[node_id] = child_ids
    
    # Convert to proper format (string keys for JSON)
    hierarchy_dict_str = {str(k): v for k, v in hierarchy_dict.items()}
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(hierarchy_dict_str, f, indent=2)
    
    print(f"Hierarchy processed and saved to {output_path}")
    return hierarchy_dict, label_to_id

def create_value_dict_and_slot(output_dir, label_to_id, hierarchy_dict):
    """Create value_dict.pt and slot.pt files for HPT"""
    
    # Create label dictionary (id -> label)
    label_dict = {v: k for k, v in label_to_id.items()}
    torch.save(label_dict, os.path.join(output_dir, "value_dict.pt"))
    
    # Create slot dictionary (parent_id -> set of child_ids)
    hierarchy = defaultdict(set)
    for parent_id, child_ids in hierarchy_dict.items():
        hierarchy[parent_id] = set(child_ids)
    
    torch.save(dict(hierarchy), os.path.join(output_dir, "slot.pt"))
    
    print(f"Created value_dict.pt and slot.pt in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert data to HPT format")
    parser.add_argument("--train", required=True, help="Path to train.json")
    parser.add_argument("--val", required=True, help="Path to val.json")
    parser.add_argument("--test", required=False, help="Path to test.json")
    parser.add_argument("--output_dir", default="data/your_dataset", help="Output directory")
    args = parser.parse_args()
    
    # Import label mapping from the fixed hierarchy
    from hierarchical_structure import LABEL_MAPPING
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Processing train data...")
    label_to_id = convert_data_format(
        args.train, 
        os.path.join(args.output_dir, "train.json"),
        label_mapping=LABEL_MAPPING
    )
    
    print("Processing validation data...")
    convert_data_format(
        args.val, 
        os.path.join(args.output_dir, "dev.json"),  # HPT uses "dev" instead of "val"
        label_to_id,
        label_mapping=LABEL_MAPPING
    )
    
    if (args.test):
        print("Processing test data...")
        convert_data_format(
            args.test, 
            os.path.join(args.output_dir, "test.json"),
            label_to_id,
            label_mapping=LABEL_MAPPING
        )
    
    # Process hierarchy tree
    print("Processing hierarchy tree...")
    hierarchy_dict, label_to_id = process_hierarchy_tree(
        os.path.join(args.output_dir, "hierarchy.json"),
        label_to_id
    )
    
    # Create HPT-specific files
    create_value_dict_and_slot(args.output_dir, label_to_id, hierarchy_dict)
    
    print(f"Data processing complete. Files saved to {args.output_dir}")
    print(f"Total labels: {len(label_to_id)}")

if __name__ == "__main__":
    main()