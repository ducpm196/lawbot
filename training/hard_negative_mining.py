#!/usr/bin/env python3
"""
Hard Negative Mining Module - LawBot v8.3
=========================================

Implements hard negative mining strategies to improve training data quality
for reranker models.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """Mines hard negative samples for reranker training."""
    
    def __init__(self, model_name: str = "vinai/phobert-base-v2"):
        """Initialize the miner with a sentence transformer model."""
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"✅ Hard negative miner initialized with {model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize hard negative miner: {e}")
            self.model = None
    
    def mine_hard_negatives(
        self,
        query: str,
        positive_docs: List[str],
        negative_candidates: List[str],
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[str]:
        """
        Mine hard negative samples based on semantic similarity.
        
        Args:
            query: The query text
            positive_docs: List of positive document texts
            negative_candidates: List of candidate negative documents
            top_k: Number of hard negatives to return
            similarity_threshold: Minimum similarity to consider as hard negative
            
        Returns:
            List of hard negative document texts
        """
        if not self.model or not negative_candidates:
            return negative_candidates[:top_k]
        
        try:
            # Encode query and documents
            query_embedding = self.model.encode([query], convert_to_tensor=True)
            doc_embeddings = self.model.encode(negative_candidates, convert_to_tensor=True)
            
            # Calculate similarities
            similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
            
            # Find documents similar to query but not positive
            hard_negatives = []
            for i, sim_score in enumerate(similarities):
                if sim_score > similarity_threshold:
                    hard_negatives.append((i, sim_score.item()))
            
            # Sort by similarity (highest first) and take top_k
            hard_negatives.sort(key=lambda x: x[1], reverse=True)
            selected_indices = [idx for idx, _ in hard_negatives[:top_k]]
            
            # If not enough hard negatives, add random ones
            if len(selected_indices) < top_k:
                remaining = list(set(range(len(negative_candidates))) - set(selected_indices)
                selected_indices.extend(np.random.choice(remaining, 
                                                      min(top_k - len(selected_indices), len(remaining)), 
                                                      replace=False))
            
            return [negative_candidates[i] for i in selected_indices[:top_k]]
            
        except Exception as e:
            logger.warning(f"Hard negative mining failed, using random selection: {e}")
            return np.random.choice(negative_candidates, 
                                  min(top_k, len(negative_candidates)), 
                                  replace=False).tolist()
    
    def create_training_triplets(
        self,
        queries: List[str],
        positive_docs: List[str],
        negative_docs: List[str],
        num_negatives_per_query: int = 3
    ) -> List[Dict[str, str]]:
        """
        Create training triplets with hard negative mining.
        
        Args:
            queries: List of query texts
            positive_docs: List of positive document texts
            negative_docs: List of negative document texts
            num_negatives_per_query: Number of negatives per query
            
        Returns:
            List of training samples with hard negatives
        """
        training_samples = []
        
        for i, (query, pos_doc) in enumerate(zip(queries, positive_docs)):
            # Mine hard negatives for this query
            hard_negatives = self.mine_hard_negatives(
                query, [pos_doc], negative_docs, 
                top_k=num_negatives_per_query
            )
            
            # Create positive sample
            training_samples.append({
                "query": query,
                "positive": pos_doc,
                "negative": hard_negatives[0] if hard_negatives else negative_docs[0],
                "label": 1.0
            })
            
            # Create negative samples
            for neg_doc in hard_negatives:
                training_samples.append({
                    "query": query,
                    "positive": pos_doc,
                    "negative": neg_doc,
                    "label": 0.0
                })
        
        logger.info(f"✅ Created {len(training_samples)} training samples with hard negatives")
        return training_samples


def enhance_training_data(
    input_file: str,
    output_file: str,
    model_name: str = "vinai/phobert-base-v2"
) -> bool:
    """
    Enhance training data with hard negative mining.
    
    Args:
        input_file: Path to input training data
        output_file: Path to output enhanced training data
        model_name: Sentence transformer model to use
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load input data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        logger.info(f"📊 Loaded {len(data)} training samples from {input_file}")
        
        # Initialize miner
        miner = HardNegativeMiner(model_name)
        
        # Extract queries, positives, and negatives
        queries = [item.get('query', '') for item in data]
        positives = [item.get('positive', item.get('passage', '')) for item in data]
        
        # Create negative candidates from all documents
        all_docs = list(set(positives))
        negative_candidates = [doc for doc in all_docs if doc not in positives[:100]]  # Limit for efficiency
        
        # Create enhanced training data
        enhanced_data = miner.create_training_triplets(
            queries, positives, negative_candidates, num_negatives_per_query=2
        )
        
        # Save enhanced data
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in enhanced_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ Enhanced training data saved to {output_file}")
        logger.info(f"📈 Data enhanced from {len(data)} to {len(enhanced_data)} samples")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to enhance training data: {e}")
        return False


if __name__ == "__main__":
    # Test the hard negative mining
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    test_queries = ["What is the penalty for tax evasion?"]
    test_positives = ["Tax evasion is punishable by law with fines and imprisonment."]
    test_negatives = [
        "The weather is sunny today.",
        "How to cook rice properly.",
        "Tax laws require proper documentation.",
        "Penalties for traffic violations."
    ]
    
    miner = HardNegativeMiner()
    hard_negs = miner.mine_hard_negatives(
        test_queries[0], test_positives, test_negatives, top_k=2
    )
    
    print(f"Query: {test_queries[0]}")
    print(f"Positive: {test_positives[0]}")
    print(f"Hard negatives: {hard_negs}")
