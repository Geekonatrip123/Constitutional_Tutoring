"""
Train Alignment Scorer (XGBoost) - Task 2.3
Predicts alignment scores from deliberation embeddings.
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_deliberations(jsonl_path: str):
    """Load deliberations from JSONL file, filtering out low-quality scores."""
    deliberations = []
    scores = []
    skipped = 0
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            score = data['alignment_score']
            
            # Filter out default/failed scores (0.5 is the fallback value)
            if score == 0.5:
                skipped += 1
                continue
            
            deliberations.append(data['deliberation_text'])
            scores.append(score)
    
    logger.info(f"Loaded {len(deliberations)} deliberations")
    logger.info(f"Skipped {skipped} deliberations with score=0.5")
    return deliberations, np.array(scores)


def create_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """Create sentence embeddings using SentenceTransformer."""
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    logger.info(f"Encoding {len(texts)} deliberations...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def train_scorer(X_train, y_train, X_val, y_val):
    """Train XGBoost regressor for alignment scoring."""
    logger.info("Training XGBoost alignment scorer...")
    
    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'early_stopping_rounds': 20
    }
    
    model = xgb.XGBRegressor(**params)
    
    # Train
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # Evaluate
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    train_mae = mean_absolute_error(y_train, train_preds)
    val_mae = mean_absolute_error(y_val, val_preds)
    train_r2 = r2_score(y_train, train_preds)
    val_r2 = r2_score(y_val, val_preds)
    
    logger.info("=" * 60)
    logger.info("TRAINING RESULTS:")
    logger.info(f"Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")
    logger.info(f"Train MAE:  {train_mae:.4f} | Val MAE:  {val_mae:.4f}")
    logger.info(f"Train R²:   {train_r2:.4f} | Val R²:   {val_r2:.4f}")
    logger.info("=" * 60)
    
    return model


def main():
    """Main training pipeline."""
    
    # Paths
    DATA_FILE = Path("../data/data/processed/synthetic_deliberations.jsonl")
    OUTPUT_DIR = Path("../../models/alignment_scorer")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("TRAINING ALIGNMENT SCORER - TASK 2.3")
    logger.info("=" * 60)
    
    # Load data (filtering out score=0.5)
    deliberations, scores = load_deliberations(DATA_FILE)
    
    # Create embeddings
    embeddings = create_embeddings(deliberations)
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, scores,
        test_size=0.2,
        random_state=42
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Val set: {len(X_val)} samples")
    
    # Train model
    model = train_scorer(X_train, y_train, X_val, y_val)
    
    # Save model
    model_path = OUTPUT_DIR / "alignment_scorer.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Save embedding model name for inference
    config = {
        'embedding_model': 'all-MiniLM-L6-v2',
        'embedding_dim': embeddings.shape[1]
    }
    config_path = OUTPUT_DIR / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to: {config_path}")
    
    logger.info("=" * 60)
    logger.info("ALIGNMENT SCORER TRAINING COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()