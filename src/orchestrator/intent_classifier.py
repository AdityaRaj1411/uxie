"""
Intent Classifier — determines retrieval parameters based on query type.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Intent:
    intent_type: str  # "precise", "exploratory", "brainstorm"
    distance_threshold: float
    max_hops: int
    semantic_weight: float
    bm25_weight: float


def get_intent_params(intent_type: str, config: dict) -> Intent:
    """
    Return the appropriate retrieval parameters from config based on intent type.
    """
    thresholds = config.get("retrieval", {}).get("thresholds", {})
    weights = config.get("retrieval", {}).get("weights", {})
    
    # Defaults
    sem_weight = weights.get("semantic", 0.5)
    bm25_weight = weights.get("bm25", 0.3)
    
    intent_type = intent_type.lower().strip()
    
    if intent_type == "precise":
        logger.debug("Intent: PRECISE")
        return Intent(
            intent_type="precise",
            distance_threshold=thresholds.get("precise", 0.65),
            max_hops=0,
            semantic_weight=sem_weight - 0.1,
            bm25_weight=bm25_weight + 0.1,
        )
        
    elif intent_type == "brainstorm":
        logger.debug("Intent: BRAINSTORM")
        return Intent(
            intent_type="brainstorm",
            distance_threshold=thresholds.get("brainstorm", 0.85),
            max_hops=2,
            semantic_weight=sem_weight + 0.1,
            bm25_weight=bm25_weight - 0.1,
        )
        
    elif intent_type == "complex_synthesis":
        logger.debug("Intent: COMPLEX_SYNTHESIS")
        return Intent(
            intent_type="complex_synthesis",
            distance_threshold=thresholds.get("brainstorm", 0.85), # Reuse brainstorm threshold
            max_hops=2,
            semantic_weight=sem_weight + 0.1,
            bm25_weight=bm25_weight,
        )
        
    else:
        logger.debug("Intent: EXPLORATORY")
        return Intent(
            intent_type="exploratory",
            distance_threshold=thresholds.get("exploratory", 0.70),
            max_hops=1,
            semantic_weight=sem_weight,
            bm25_weight=bm25_weight,
        )
