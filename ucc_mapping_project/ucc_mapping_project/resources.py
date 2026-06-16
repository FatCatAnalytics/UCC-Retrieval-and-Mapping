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
        default=str(_PROJECT_ROOT / "Inputs" / "ucc.csv"),
        description="Path to UCC data file"
    )
    master_file: str = Field(
        default=str(_PROJECT_ROOT / "Inputs" / "Master Mapping.csv"),
        description="Path to master mapping file"
    )
    universe_file: str = Field(
        default=str(_PROJECT_ROOT / "Inputs" / "Universe.csv"),
        description="Path to tracked banks file containing Bank Name and Firm ID"
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

    # Lead-bank designation (Stage 8)
    lead_bank_threshold: float = Field(
        default=0.66,
        description="Minimum share of total filings (originals + continuations) a single bank must hold to be designated the lead bank for a company. Default 0.66 = 66%."
    )
    filing_type_column: str = Field(
        default="",
        description="Override: name of the raw-CSV column that holds filing type (originals vs continuations). Leave blank to auto-detect from common names (FILING_TYPE, FILING_TYP, RECORD_TYPE, TYPE, FILING)."
    )
    debtor_company_column: str = Field(
        default="",
        description="Override: name of the raw-CSV column that holds the debtor / company name. Leave blank to auto-detect from common names (DEBT_NAME, DEBTOR_NAME, DEBT_FIRM, DEBTOR, COMPANY_NAME, DEBTOR_COMPANY)."
    )

    # Clustering performance
    cluster_memmap_threshold_mb: int = Field(default=512, description="Use disk-backed embeddings above this size")
    cluster_hnsw_threshold: int = Field(default=200000, description="Use HNSW for large clustering")
    cluster_hnsw_m: int = Field(default=32, description="HNSW M parameter")
    cluster_hnsw_ef_search: int = Field(default=128, description="HNSW efSearch parameter")
    cluster_hnsw_ef_construction: int = Field(default=200, description="HNSW efConstruction parameter")

    # Stage 4 advanced-match performance / accuracy upgrades
    advanced_use_per_state_faiss: bool = Field(
        default=True,
        description="Build a FAISS IndexFlatIP per state-pair in master_embeddings_cache and "
                    "use it in Stage 4 instead of repeated np.matmul calls. Mathematically "
                    "identical to the matmul path; ~20-30% faster wall-clock at scale."
    )
    advanced_per_state_min_size: int = Field(
        default=64,
        description="State-pair sizes below this stay on the legacy matmul path — building a "
                    "FAISS index for tiny groups is more overhead than the search saves."
    )
    query_embed_cache_enabled: bool = Field(
        default=True,
        description="Persist (model, query_norm) -> embedding to disk across runs so re-running "
                    "Stage 4 on the same UCC inputs skips the SentenceTransformer encode step."
    )
    query_embed_cache_path: str = Field(
        default=str(_PROJECT_ROOT / ".cache" / "query_embeddings"),
        description="Directory for the persistent query embedding cache. One .npy + one .json "
                    "shard per embedding model id (sanitised)."
    )
    # Cross-encoder rerank for REVIEW_MEDIUM band
    advanced_cross_encoder_enabled: bool = Field(
        default=True,
        description="After Stage 4 selects a match, re-score REVIEW_MEDIUM rows with a small "
                    "cross-encoder over the top candidates. May promote matches to REVIEW_HIGH "
                    "or AUTO (never demotes). Safe to leave on — turns into a no-op if the "
                    "cross-encoder model is not available locally."
    )
    advanced_cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="HuggingFace model id (or local path) for the rerank cross-encoder. The "
                    "ms-marco-MiniLM-L-6 default is small (~80MB) and CPU-friendly."
    )
    advanced_cross_encoder_top_n: int = Field(
        default=10,
        description="Number of top candidates per REVIEW_MEDIUM query to feed the cross-encoder."
    )
    advanced_cross_encoder_promote_to_auto: float = Field(
        default=0.92,
        description="Cross-encoder relevance score above which a REVIEW_MEDIUM match is promoted "
                    "to AUTO. Cross-encoder outputs are typically in [-10, +10] passed through "
                    "a sigmoid; we read the sigmoid score (0..1)."
    )
    advanced_cross_encoder_promote_to_high: float = Field(
        default=0.80,
        description="Cross-encoder score above which a REVIEW_MEDIUM match is promoted to "
                    "REVIEW_HIGH. Must be < promote_to_auto."
    )
