from pycleora import SparseMatrix
import numpy as np
import os

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

def run_cleora():
    hyperedge_path = os.path.join(_DATA_DIR, 'hyperedges_cleora.txt')
    if not os.path.exists(hyperedge_path):
        print(f"Error: {hyperedge_path} not found.")
        return

    print(f"Loading hyperedges from {hyperedge_path}...")
    with open(hyperedge_path, 'r', encoding='utf-8') as f:
        # cleora SparseMatrix.from_iterator expects an iterator of strings
        # each string is a space-separated list of entities in a hyperedge
        mat = SparseMatrix.from_iterator(f, columns='complex::reflexive::product')

    print(f"Entities found: {len(mat.entity_ids)}")
    
    # Initialize embeddings
    DIM = 1024
    print(f"Initializing embeddings with dim={DIM}...")
    embeddings = mat.initialize_deterministically(DIM)

    # Markov propagation
    NUM_WALKS = 8
    print(f"Running {NUM_WALKS} Markov propagation steps for deep behavioral discovery...")
    for i in range(NUM_WALKS):
        print(f"Walk {i+1}/{NUM_WALKS}...")
        embeddings = mat.left_markov_propagate(embeddings)
        # L2 normalization
        embeddings /= np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)

    # Save embeddings
    output_path = os.path.join(_DATA_DIR, 'cleora_embeddings.npz')
    np.savez(output_path, asins=mat.entity_ids, embeddings=embeddings)
    print(f"Saved Cleora embeddings for {len(mat.entity_ids)} items to {output_path}")

if __name__ == "__main__":
    run_cleora()
