"""
ECO-EDGE: SLM Configuration & Provider
========================================
Configures the connection to a local Ollama instance and provides
the LangChain LLM object used by the agents.
"""

from __future__ import annotations
import os
import logging
from typing import Optional

from langchain_ollama import OllamaLLM

# Configuration
def _get_config():
    return {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "model": os.getenv("OLLAMA_MODEL", "phi3:mini"),
        "use_slm": os.getenv("USE_SLM", "false").lower() == "true"
    }

logger = logging.getLogger("eco_edge.slm")

def get_slm(model_name: Optional[str] = None, temperature: float = 0.1) -> Optional[OllamaLLM]:
    """
    Returns a configured OllamaLLM instance if USE_SLM is True.
    Returns None if SLM is disabled (falling back to rule-based logic).
    """
    config = _get_config()
    if not config["use_slm"]:
        return None
    
    try:
        llm = OllamaLLM(
            model=model_name or config["model"],
            base_url=config["base_url"],
            temperature=temperature,
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to connect to Ollama at {config['base_url']}: {e}")
        return None
