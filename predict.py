import torch
import argparse
import os
import datasets
from models.prompt import Prompt
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json
from tqdm import tqdm
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints/memes-subtask1-hpt/checkpoint_last.pt")
    parser.add_argument('--test_data', type=str, default='data/proc_datas/en_subtask1_test_unlabeled.json')
    parser.add_argument('--data', type=str, default='data/proc_datas/')
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--graph', type=str, default='GAT')
    parser.add_argument('--arch', type=str, default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model', type=str, default='prompt')
    parser.add_argument('--output_dir', type=str, default='predictions/subtask1/')
    return parser

def format_predictions(predictions, ids, label_list):
    """Format predictions for output"""
    results = []
    print(label_list)
    for i, pred in enumerate(predictions):
        print(pred)
        labels = [label_list[idx] for idx in pred]
        results.append({
            "id": ids[i],
            "labels": labels
        })
    
    return results

def load_model_and_predict(device='cuda' if torch.cuda.is_available() else 'cpu'):
    args = parse().parse_args()
    data_path = args.data
    
    label_dict = torch.load(os.path.join(data_path, 'value_dict.pt'))
    label_dict = {i: v for i, v in label_dict.items()}
    label_list = [label_dict[i] for i in range(len(label_dict))]
    # slot2value = torch.load(os.path.join(data_path, 'slot.pt'), weights_only=False)
    slot2value = torch.load(os.path.join(data_path, 'slot.pt'))
    
    value2slot = {}
    num_class = 0

    for s in slot2value:
        for v in slot2value[s]:
            value2slot[v] = s
            if num_class < v:
                num_class = v
    num_class += 1

    path_list = [(i, v) for v, i in value2slot.items()]
    for i in range(num_class):
        if i not in value2slot:
            value2slot[i] = -1

    def get_depth(x):
        depth = 0
        while value2slot[x] != -1:
            depth += 1
            x = value2slot[x]
        return depth


    depth_dict = {i: get_depth(i) for i in range(num_class)}
    max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
    depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}
    print(depth2label)
    
    for depth in depth2label:
        for l in depth2label[depth]:
            path_list.append((num_class + depth, l))


    model = Prompt.from_pretrained(args.arch, num_labels=len(label_dict), path_list=path_list, layer=args.layer,
                                   graph_type=args.graph, data_path=data_path, depth2label=depth2label,)
    tokenizer = AutoTokenizer.from_pretrained(args.arch)
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model.init_embedding()
    model.load_state_dict(checkpoint['param'])
    model.to('cuda')
    predictions = []
    
    if args.model == 'prompt':
        dataset = datasets.load_dataset('json',
                                            data_files={'test': 'data/proc_datas/en_subtask1_test_unlabeled.json'.format(args.data, args.data), 
                                                        })

        prefix = []
        for i in range(max_depth):
            prefix.append(tokenizer.vocab_size + num_class + i)
            prefix.append(tokenizer.vocab_size + num_class + max_depth)
        prefix.append(tokenizer.sep_token_id)

        
        def data_map_function(batch, tokenizer):
            new_batch = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
            for t in batch['text']:
                tokens = tokenizer(t, truncation=True)
                new_batch['input_ids'].append(tokens['input_ids'][:-1][:512 - len(prefix)] + prefix)
                new_batch['input_ids'][-1].extend(
                    [tokenizer.pad_token_id] * (512 - len(new_batch['input_ids'][-1])))
                new_batch['attention_mask'].append(
                    tokens['attention_mask'][:-1][:512 - len(prefix)] + [1] * len(prefix))
                new_batch['attention_mask'][-1].extend([0] * (512 - len(new_batch['attention_mask'][-1])))
                new_batch['token_type_ids'].append([0] * 512)

            return new_batch

        dataset = dataset.map(lambda x: data_map_function(x, tokenizer), batched=True)
        dataset.save_to_disk(os.path.join(data_path, args.model))
        dataset['test'].set_format('torch', columns=['attention_mask', 'input_ids'])
        print(dataset['test'])

    test = DataLoader(dataset['test'], batch_size=8, shuffle=False)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(test):
            batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            output_ids, logits = model.generate(batch['input_ids'], depth2label=depth2label, )
            for labels_list in output_ids:
                unique_ids = set()

                for label in labels_list:
                    unique_ids.add(label)
                predictions.append(list(unique_ids))
                
    #format predictions
    orig_dataset = datasets.load_dataset('json',
                                            data_files={'test': 'data/proc_datas/en_subtask1_test_unlabeled.json'
                                                        })
    ids = [item['id'] for item in orig_dataset['test']]
    formatted_predictions = format_predictions(predictions, ids, label_list)
    # Save predictions to a file
    output_file = os.path.join(args.output_dir, "predictions.txt")

    with open(output_file, 'w') as f:  
        json.dump(formatted_predictions, f, indent=2)
    print(f"Predictions saved to {output_file}")
    return 

# Example usage
if __name__ == "__main__":
    load_model_and_predict()
    