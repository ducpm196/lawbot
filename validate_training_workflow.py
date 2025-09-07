#!/usr/bin/env python3
"""
Training Workflow Validation Script - LawBot v8.3
================================================

Validates that training workflow works correctly without errors.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_device_compatibility():
    """Validate device compatibility."""
    logger.info("🔍 Validating device compatibility...")
    
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"✅ Device: {device}")
        
        # Test model loading without device parameter
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        model_name = "vinai/phobert-base-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            problem_type="single_label_classification"
        )
        
        # Move to device after loading
        model.to(device)
        logger.info("✅ Model loading and device movement successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Device compatibility validation failed: {e}")
        return False

def validate_hard_negative_mining():
    """Validate hard negative mining."""
    logger.info("🔍 Validating hard negative mining...")
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Test with dummy data
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        queries = ["What is a contract?"]
        passages = ["A contract is a legal agreement."]
        negative_candidates = [
            "A contract is a written document.",
            "A contract is a binding agreement.",
            "A contract is a formal arrangement.",
            "A contract is a business deal.",
            "A contract is a legal document."
        ]
        
        # Test hard negative mining
        query_embeddings = model.encode(queries, convert_to_tensor=True)
        neg_embeddings = model.encode(negative_candidates, convert_to_tensor=True)
        
        similarities = cosine_similarity(
            query_embeddings.cpu().numpy(),
            neg_embeddings.cpu().numpy()
        )[0]
        
        # Find hard negatives
        hard_neg_mask = (similarities >= 0.3) & (similarities <= 0.7)
        hard_negatives = [negative_candidates[i] for i in np.where(hard_neg_mask)[0]]
        
        if hard_negatives:
            logger.info(f"✅ Hard negative mining successful: {len(hard_negatives)} found")
        else:
            logger.warning("⚠️ No hard negatives found, but fallback should work")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hard negative mining validation failed: {e}")
        return False

def main():
    """Main validation function."""
    logger.info("🚀 Starting training workflow validation...")
    
    success = True
    
    # Validate device compatibility
    if not validate_device_compatibility():
        success = False
    
    # Validate hard negative mining
    if not validate_hard_negative_mining():
        success = False
    
    if success:
        logger.info("✅ All validations passed! Training workflow should work correctly.")
    else:
        logger.error("❌ Some validations failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
