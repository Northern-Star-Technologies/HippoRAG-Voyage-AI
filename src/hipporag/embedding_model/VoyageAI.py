"""
Voyage AI Embedding Model for HippoRAG

Integrates Voyage AI's embedding and reranking APIs with HippoRAG's architecture.
Supports voyage-3, voyage-3-lite, and voyage-code-3 models.
"""

from copy import deepcopy
from typing import List, Optional
import numpy as np
from tqdm import tqdm

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)

# Optional Voyage AI import with graceful fallback
try:
    import voyageai
    VOYAGE_AVAILABLE = True
except ImportError:
    VOYAGE_AVAILABLE = False
    logger.warning("voyageai package not installed. Install with: pip install voyageai")


class VoyageAIEmbeddingModel(BaseEmbeddingModel):
    """
    Voyage AI embedding model implementation for HippoRAG.

    Supports:
    - voyage-3: Best quality, 1024 dimensions
    - voyage-3-lite: 3x cheaper, 512 dimensions
    - voyage-code-3: Code-optimized, 1024 dimensions

    Features:
    - Asymmetric query/document embeddings via input_type
    - 320K token context window (voyage-3)
    - Built-in normalization
    """

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if not VOYAGE_AVAILABLE:
            raise ImportError(
                "voyageai package is required for VoyageAI embeddings. "
                "Install it with: pip install voyageai"
            )

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()

        # Initialize Voyage AI client
        logger.debug(
            f"Initializing {self.__class__.__name__}'s embedding model: {self.embedding_model_name}")

        # Get API key from environment or config
        import os
        api_key = os.getenv("VOYAGEAI_API_KEY")
        if not api_key:
            raise ValueError(
                "VOYAGEAI_API_KEY environment variable not set. "
                "Get your API key from: https://dash.voyageai.com/"
            )

        self.client = voyageai.Client(api_key=api_key, max_retries=3)

        # Determine embedding dimensions based on model
        if "lite" in self.embedding_model_name:
            self.embedding_dim = 512
        elif "code" in self.embedding_model_name:
            self.embedding_dim = 1024
        else:  # voyage-3, voyage-3.5
            self.embedding_dim = 1024

        logger.info(f"✅ Voyage AI embeddings initialized: {self.embedding_model_name} (dim={self.embedding_dim})")

    def _init_embedding_config(self) -> None:
        """
        Initialize embedding configuration compatible with HippoRAG's expectations.
        """
        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized if self.global_config else True,
            "model_init_params": {
                "pretrained_model_name_or_path": self.embedding_model_name,
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len if self.global_config else 2048,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size if self.global_config else 16,
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def encode(self, texts: List[str], input_type: str = "document") -> np.ndarray:
        """
        Encode texts using Voyage AI API.

        Args:
            texts: List of text strings to embed
            input_type: "query" or "document" (critical for retrieval accuracy)

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        # Clean texts (replace newlines, handle empty strings)
        texts = [t.replace("\n", " ") for t in texts]
        texts = [t if t != '' else ' ' for t in texts]

        # Call Voyage AI API
        response = self.client.embed(
            texts=texts,
            model=self.embedding_model_name,
            input_type=input_type,
            truncation=True
        )

        # Convert to numpy array
        results = np.array(response.embeddings)

        return results

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Batch encode texts using Voyage AI embeddings.

        Args:
            texts: List of text strings to embed
            **kwargs: Additional parameters
                - instruction: Query instruction (maps to input_type)
                - batch_size: Batch size for API calls
                - norm: Whether to normalize embeddings

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([])

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")

        # Map HippoRAG instruction to Voyage AI input_type
        # HippoRAG uses instructions like "query_to_fact", "query_to_passage"
        instruction = params.get("instruction", "")
        if "query" in instruction.lower():
            input_type = "query"
        else:
            input_type = "document"

        logger.debug(f"Using Voyage AI input_type={input_type} (instruction: {instruction})")

        # Batch processing
        batch_size = params.get("batch_size", 16)

        if len(texts) <= batch_size:
            results = self.encode(texts, input_type=input_type)
        else:
            # Process in batches with progress bar
            pbar = tqdm(total=len(texts), desc=f"Voyage AI Encoding ({input_type})")
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    batch_results = self.encode(batch, input_type=input_type)
                    results.append(batch_results)
                except Exception as e:
                    logger.error(f"Batch encoding failed: {e}")
                    raise
                pbar.update(len(batch))
            pbar.close()
            results = np.concatenate(results)

        # Normalization (Voyage AI returns normalized by default, but double-check)
        if self.embedding_config.norm and not self._is_normalized(results):
            logger.debug("Normalizing embeddings")
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results

    def _is_normalized(self, embeddings: np.ndarray, tolerance: float = 1e-5) -> bool:
        """Check if embeddings are already normalized."""
        norms = np.linalg.norm(embeddings, axis=1)
        return np.allclose(norms, 1.0, atol=tolerance)
