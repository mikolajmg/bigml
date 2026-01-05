import sys
import pandas as pd
import json
import numpy as np
from collections import Counter
import re
from mpi4py import MPI 

def predict_tree(node, features_present):
    if isinstance(node, (int, np.integer)):
        return node
    
    
    feature_idx, left_subtree, right_subtree = node
    
    
    if feature_idx in features_present:
        return predict_tree(right_subtree, features_present)
    else:
        return predict_tree(left_subtree, features_present)

def main():
    print("Starting classification...")
    if len(sys.argv) < 4:
        print("Usage: mpirun -n <n> python classify.py <model_input> <query_input> <predictions_output>")
        return

    model_input_base = sys.argv[1]
    query_input = sys.argv[2]
    predictions_output = sys.argv[3]

    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # rank = 0 
    # size = 1


    local_model_path = f"{model_input_base}_{rank}.txt"
    with open(local_model_path, 'r') as f:
        lines = f.readlines()

        local_vocab = lines[0].strip().split()
        word_to_idx = {word: i for i, word in enumerate(local_vocab)}

        local_trees = [json.loads(line.strip()) for line in lines[1:] if line.strip()]

    queries = pd.read_csv(query_input, header=None)[0]
    
    local_results = []
    for query in queries:
        clean_query = str(query).lower()
        
        clean_query = re.sub(r'[^a-z\s]', '', clean_query)
        words_in_query = set(clean_query.split())
        
        
        features_present = {word_to_idx[w] for w in words_in_query if w in word_to_idx}
        
        
        votes = [predict_tree(tree, features_present) for tree in local_trees]
        local_results.append(votes)

    
    local_results = np.array(local_results) 

    
    all_votes = comm.gather(local_results, root=0)
    all_votes = [local_results] # Mock for local testing

    if rank == 0:
        final_predictions = []
        combined_votes = np.hstack(all_votes)

        for query_votes in combined_votes:
            counts = Counter(query_votes)
            max_vote_count = max(counts.values())
            winners = [cls for cls, count in counts.items() if count == max_vote_count]
            final_predictions.append(min(winners))

        with open(predictions_output, 'w') as f:
            for pred in final_predictions:
                f.write(f"{pred}\n")

if __name__ == "__main__":
    main()