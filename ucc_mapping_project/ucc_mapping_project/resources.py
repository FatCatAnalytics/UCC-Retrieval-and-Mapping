# =============================================================================
# RESOURCES
# =============================================================================
import os
from pathlib import Path

from dagster import ConfigurableResource, EnvVar

from pydantic import Field

# Compute project root dynamically (works regardless of working directory)
_PROJECT_ROOT = Path(__file__).parent


class PipelineConfig(ConfigurableResource):
    """Central configuration for the mapping pipeline."""
    
    # File paths — use absolute paths derived from project structure
    ucc_file: str = Field(
        default=str(_PROJECT_ROOT / "Inputs" / "raw.csv"),
        description="Path to UCC data file"
    )
    master_file: str = Field(
        default=str(_PROJECT_ROOT / "Inputs" / "master_list.csv"),
        description="Path to master mapping file"
    )
    universe_file: str = Field(
        default=str(_PROJECT_ROOT / "Inputs" / "our universe.csv"),
        description="Path to our-universe file (tab-delimited) containing SECP_NAME, GU_COMPANY_NAME, and Firm_ID"
    )
    output_dir: str = Field(
        default=str(_PROJECT_ROOT / "Outputs"),
        description="Directory for output files"
    )

    # Model path — use local model if present, otherwise HuggingFace will auto-download
    embedding_model: str = Field(
        default=str(_PROJECT_ROOT / "models" / "bge-small-en-v1.5"),
        description="Path to local embedding model directory, or a HuggingFace model id (e.g. 'BAAI/bge-small-en-v1.5')"
    )
    
    # Matching thresholds
    fuzzy_auto_threshold: int = Field(default=96, description="Auto-approve fuzzy matches >= this (tightened)")
    fuzzy_review_threshold: int = Field(default=85, description="Review fuzzy matches >= this (wider net for review)")
    fuzzy_min_threshold: int = Field(default=80, description="Minimum fuzzy score to consider")
    
    embed_auto_threshold: float = Field(default=0.95, description="Auto-approve embedding matches >= this (tightened)")
    embed_high_threshold: float = Field(default=0.90, description="High confidence review threshold")
    embed_medium_threshold: float = Field(default=0.85, description="Medium confidence review threshold")
    embed_top_k: int = Field(default=20, description="Top K candidates to consider for embedding matching")
    
    # Clustering parameters
    cluster_sim_threshold: float = Field(default=0.88, description="Similarity for initial clustering")
    cluster_merge_threshold: float = Field(default=0.88, description="Threshold for merging similar clusters - RAISED from 0.70")
    cluster_k_neighbors: int = Field(default=20, description="Neighbors to consider in clustering")
    cluster_fuzzy_guard: int = Field(default=80, description="Fuzzy ratio guard for cluster edges")
    cluster_merge_iterations: int = Field(default=1, description="Number of merge iterations - REDUCED from 3")
    cluster_merge_k: int = Field(default=10, description="Neighbors to check during merge - REDUCED from 25")
    cluster_max_size: int = Field(default=500, description="Maximum cluster size to prevent mega-clusters")
    
    # Processing parameters - OPTIMIZED for speed
    embed_batch_size: int = Field(default=256, description="Batch size for embedding")
    search_batch_size: int = Field(default=4096, description="Batch size for search")
    fuzzy_batch_size: int = Field(default=1024, description="Batch size for fuzzy matching (increased for parallel processing)")
    fuzzy_workers: int = Field(default=-1, description="Number of workers for fuzzy matching (-1 = all cores)")

    # CPU threading optimization
    cpu_threads: int = Field(default=-1, description="Number of CPU threads (-1 = auto)")
    embedding_workers: int = Field(default=4, description="Parallel workers for embedding")

    # Early termination settings
    early_stop_threshold: float = Field(default=0.98, description="Stop searching after finding match this good")
    max_candidates_per_query: int = Field(default=50, description="Maximum candidates to evaluate per query")

    # Clustering performance
    cluster_memmap_threshold_mb: int = Field(default=512, description="Use disk-backed embeddings above this size")
    cluster_hnsw_threshold: int = Field(default=200000, description="Use HNSW for large clustering")
    cluster_hnsw_m: int = Field(default=32, description="HNSW M parameter")
    cluster_hnsw_ef_search: int = Field(default=128, description="HNSW efSearch parameter")
    cluster_hnsw_ef_construction: int = Field(default=200, description="HNSW efConstruction parameter")
