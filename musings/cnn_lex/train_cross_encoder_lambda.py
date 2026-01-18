"""
Train cross-encoder on MS MARCO v1.1 using sentence-transformers' LambdaLoss.

This closely follows the official training script:
https://github.com/huggingface/sentence-transformers/blob/main/examples/cross_encoder/training/ms_marco/training_ms_marco_lambda_hard_neg.py

Key: Uses LambdaLoss (learning-to-rank) instead of NCE (classification).
"""

import logging
import torch
import numpy as np
import random
import gc
import faiss
from datetime import datetime
from datasets import Dataset, load_dataset
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.losses import LambdaLoss, NDCGLoss2PPScheme
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def prepare_data(max_examples=None, num_negatives=9, skip_n_hardest=3):
    """
    Prepare MS MARCO v1.1 with mined hard negatives.
    Returns HuggingFace Dataset with 'query', 'docs', 'labels' columns.
    """
    logging.info("Loading MS MARCO v1.1 train set...")
    raw_dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")
    
    if max_examples:
        raw_dataset = raw_dataset.select(range(min(max_examples, len(raw_dataset))))
    
    logging.info(f"Loaded {len(raw_dataset)} examples")
    
    # ==========================================================================
    # Part 1: Bing hard negatives (listwise dataset)
    # ==========================================================================
    logging.info("Creating listwise dataset from Bing passages...")
    listwise_queries = []
    listwise_docs = []
    listwise_labels = []
    
    for item in tqdm(raw_dataset, desc="Procssing Bing passages"):
        query = item["query"]
        passages = item["passages"]["passage_text"]
        labels = item["passages"]["is_selected"]
        
        # Sort by label (positives first)
        paired = sorted(zip(passages, labels), key=lambda x: x[1], reverse=True)
        if not paired:
            continue
            
        sorted_passages, sorted_labels = zip(*paired)
        
        # Skip queries without positives
        if max(sorted_labels) < 1.0:
            continue
        
        listwise_queries.append(query)
        listwise_docs.append(list(sorted_passages))
        listwise_labels.append(list(sorted_labels))
    
    logging.info(f"Created {len(listwise_queries)} listwise examples from Bing passages")
    
    # ==========================================================================
    # Part 2: Mine additional hard negatives
    # ==========================================================================
    logging.info("Mining additional hard negatives...")
    
    # Extract query-positive pairs
    queries = []
    positives = []
    for item in raw_dataset:
        query = item["query"]
        passages = item["passages"]["passage_text"]
        labels = item["passages"]["is_selected"]
        for passage, label in zip(passages, labels):
            if label > 0:
                queries.append(query)
                positives.append(passage)
    
    logging.info(f"Created {len(queries):,} query-positive pairs")
    
    # Build corpus
    all_passages = []
    for item in raw_dataset:
        all_passages.extend(item["passages"]["passage_text"])
    all_passages = list(set(all_passages))
    logging.info(f"Corpus contains {len(all_passages):,} unique passages")
    
    # Encode with bi-encoder (GPU)
    logging.info("Loading bi-encoder for mining...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    logging.info("Encoding corpus on GPU...")
    corpus_embeddings = embedding_model.encode(
        all_passages,
        batch_size=512,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    
    logging.info("Encoding queries on GPU...")
    query_embeddings = embedding_model.encode(
        queries,
        batch_size=512,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    
    del embedding_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # FAISS search (CPU)
    logging.info("Building FAISS index on CPU...")
    dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_embeddings.astype(np.float32))
    
    logging.info("Searching for hard negatives...")
    k_search = skip_n_hardest + num_negatives * 3
    
    all_indices = []
    for start in tqdm(range(0, len(query_embeddings), 5000), desc="FAISS search"):
        end = min(start + 5000, len(query_embeddings))
        batch = query_embeddings[start:end].astype(np.float32)
        _, batch_indices = index.search(batch, k_search)
        all_indices.append(batch_indices)
    
    indices = np.vstack(all_indices)
    del corpus_embeddings, query_embeddings, index, all_indices
    gc.collect()
    
    # Build mined examples
    logging.info("Building mined examples...")
    mined_queries = []
    mined_docs = []
    mined_labels = []
    
    for i in tqdm(range(len(queries)), desc="Building mined examples"):
        query = queries[i]
        positive = positives[i]
        
        hard_negatives = []
        for j in range(skip_n_hardest, k_search):
            idx = indices[i][j]
            passage = all_passages[idx]
            if passage != positive and len(hard_negatives) < num_negatives:
                hard_negatives.append(passage)
        
        if len(hard_negatives) > 0:
            while len(hard_negatives) < num_negatives:
                hard_negatives.append(random.choice(hard_negatives))
            
            mined_queries.append(query)
            mined_docs.append([positive] + hard_negatives[:num_negatives])
            mined_labels.append([1] + [0] * num_negatives)
    
    logging.info(f"Mined {len(mined_queries)} examples")
    
    # Combine
    all_queries = listwise_queries + mined_queries
    all_docs = listwise_docs + mined_docs
    all_labels = listwise_labels + mined_labels
    
    logging.info(f"Total: {len(all_queries)} examples")
    
    # Create HF Dataset
    dataset = Dataset.from_dict({
        "query": all_queries,
        "docs": all_docs,
        "labels": all_labels,
    })
    
    return dataset


def main():
    # Config - matching TinyBERT training
    model_name = "nreimers/BERT-Tiny_L-2_H-128_A-2"
    train_batch_size = 16
    eval_batch_size = 16
    mini_batch_size = 16
    num_epochs = 1
    
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 1. Initialize CrossEncoder
    logging.info(f"Loading model: {model_name}")
    torch.manual_seed(12)
    model = CrossEncoder(model_name, num_labels=1)
    logging.info(f"Model max length: {model.max_length}")
    
    # 2. Prepare data
    dataset = prepare_data(num_negatives=9, skip_n_hardest=3)
    
    # Split
    dataset = dataset.train_test_split(test_size=1000, seed=12)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    logging.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # 3. Define loss - LambdaLoss for learning-to-rank
    loss = LambdaLoss(
        model=model,
        weighting_scheme=NDCGLoss2PPScheme(),
        mini_batch_size=mini_batch_size,
    )
    
    # 4. Evaluator
    evaluator = CrossEncoderNanoBEIREvaluator(
        dataset_names=["msmarco", "nfcorpus", "nq"],
        batch_size=eval_batch_size
    )
    evaluator(model)  # Initial eval
    
    # 5. Training arguments
    run_name = f"tinybert-msmarco-lambdaloss-{dt}"
    args = CrossEncoderTrainingArguments(
        output_dir=f"models/{run_name}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_R100_mean_ndcg@10",
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=250,
        logging_first_step=True,
        run_name=run_name,
        seed=12,
    )
    
    # 6. Train
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()
    
    # 7. Final eval
    evaluator(model)
    
    # 8. Save
    final_dir = f"models/{run_name}/final"
    model.save_pretrained(final_dir)
    logging.info(f"Saved to {final_dir}")


if __name__ == "__main__":
    main()
