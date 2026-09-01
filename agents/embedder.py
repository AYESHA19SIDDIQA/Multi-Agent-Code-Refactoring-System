import os
import json
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import faiss
import numpy as np 
import torch

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.eval()

def generate_embeddings(parsed_file="parsed_repo.json", metadata_file="code_metadata.json"):
    # Load parsed repo chunks
    with open(parsed_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = []
    metadata = []

    for chunk in tqdm(chunks, desc="Generating embeddings"):
        text = chunk["code"]
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**tokens)

            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)
    return embeddings

def store_embeddings(embeddings, index_file="code_embeddings.faiss"):
    # Convert to numpy array for FAISS
    embedding_matrix = np.stack(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    faiss.write_index(index, index_file)
    print(f"Saved FAISS index to '{index_file}'")
