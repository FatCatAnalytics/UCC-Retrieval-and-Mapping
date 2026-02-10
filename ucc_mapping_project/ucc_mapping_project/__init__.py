from .definitions import defs
from .utils import (
    soundex,
    detect_entity_type,
    generate_ngrams,
    ngram_similarity,
    tfidf_ngram_similarity,
    has_shared_meaningful_identifier,
    extract_entity_core,
    combined_fuzzy_score,
    normalize_name,
    min_token_overlap,
    entity_types_compatible,
    should_merge_clusters,
)
from .models.data_structures import NGramIndex, TFIDFNGramIndex, UnionFind
from .resources import PipelineConfig

