
# =============================================================================
# DAGSTER ASSETS
# =============================================================================
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
from dagster import asset, AssetExecutionContext, Output, MetadataValue, MaterializeResult
import gc
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # macOS Accelerate framework
os.environ['NUMEXPR_NUM_THREADS'] = '1'
from datetime import datetime
from .utils import (
    soundex,
    detect_entity_type,
    normalize_name,
    generate_ngrams,
    get_meaningful_words,
    ngram_similarity,
    tfidf_ngram_similarity,
    has_shared_meaningful_identifier,
    extract_entity_core,
    extract_acronym,
    is_acronym_match,
    compute_match_score,
    combined_fuzzy_score,
    min_token_overlap,
    entity_types_compatible,
    should_merge_clusters
)
from .models.data_structures import NGramIndex, UnionFind, TFIDFNGramIndex
from .resources import PipelineConfig


STATE_COLUMNS = ["DEBT_STATE", "SECP_STATE"]
RAW_MATCH_COLUMNS = STATE_COLUMNS + ["SECP_NAME"]
STATE_MISSING_SENTINEL = "__MISSING_STATE__"
NAME_MISSING_SENTINEL = "__MISSING_NAME__"
NO_MASTER_STATE_PAIR = "NO_MASTER_STATE_PAIR"


def _standardize_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize state columns so state-aware joins use stable keys."""
    df = df.copy()
    for col in STATE_COLUMNS:
        if col in df.columns:
            state = df[col].astype("string").str.upper().str.strip()
            df[col] = state.mask(state.eq(""))
    return df


def _clean_key_component(series: pd.Series, *, uppercase: bool = True) -> pd.Series:
    values = series.astype("string").str.strip()
    if uppercase:
        values = values.str.upper()
    return values.mask(values.eq("")).fillna(NAME_MISSING_SENTINEL)


def _state_pair_keys(df: pd.DataFrame) -> pd.Series:
    debt = _clean_key_component(df["DEBT_STATE"], uppercase=True).replace(
        NAME_MISSING_SENTINEL, STATE_MISSING_SENTINEL
    )
    secp = _clean_key_component(df["SECP_STATE"], uppercase=True).replace(
        NAME_MISSING_SENTINEL, STATE_MISSING_SENTINEL
    )
    return pd.Series(list(zip(debt, secp)), index=df.index)


def _state_name_keys(
    df: pd.DataFrame,
    name_col: str,
    *,
    normalized: bool = False,
) -> pd.Series:
    state_pairs = _state_pair_keys(df)
    if normalized:
        names = df[name_col].apply(normalize_name).astype("string")
    else:
        names = _clean_key_component(df[name_col], uppercase=True)
    names = names.mask(names.str.strip().eq("")).fillna(NAME_MISSING_SENTINEL)
    return pd.Series(
        [(debt, secp, name) for (debt, secp), name in zip(state_pairs, names)],
        index=df.index,
    )


def _unique_candidates(series: pd.Series) -> List[str]:
    candidates = []
    for value in series:
        if value is None or pd.isna(value):
            continue
        candidate = str(value).strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _preview_markdown(df: pd.DataFrame, rows: int = 5) -> str:
    """Render Dagster preview metadata without exposing tabulate to pandas NA."""
    return df.head(rows).astype("object").where(pd.notna(df.head(rows)), "").to_markdown()


def _detect_delimiter(file_path: str) -> str:
    """Detect comma vs tab-delimited input from the header row."""
    with open(file_path, "r", encoding="latin-1") as f:
        header = f.readline()
    return "\t" if header.count("\t") > header.count(",") else ","


def _pending_match_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df['SECP_NAME_MAPPED'].isna()
        & df['SECP_NAME_NORMALIZED'].notna()
        & (df['SECP_NAME_NORMALIZED'].str.len() > 0)
        & (df['confidence'].fillna("") != NO_MASTER_STATE_PAIR)
    )


@asset(
    description="Load raw UCC filing data from CSV",
    compute_kind="pandas",
    group_name="ingestion",
    owners=["team:data-engineering"],
    tags={"domain": "ucc-mapping", "stage": "ingestion"},
)
def ucc_raw_data(context: AssetExecutionContext, config: PipelineConfig) -> Output[pd.DataFrame]:
    """Load and deduplicate UCC filing data."""
    df = pd.read_csv(
        config.ucc_file,
        sep=_detect_delimiter(config.ucc_file),
        encoding="latin-1",
        usecols=RAW_MATCH_COLUMNS,
    ).drop_duplicates()
    df = _standardize_state_columns(df)
    
    df = df[df['SECP_NAME'].notna() & (df['SECP_NAME'].str.strip() != '')]
    df = df.reset_index(drop=True)
    
    context.log.info(f"Loaded {len(df):,} unique UCC records")
    
    # Return Output with metadata facets
    return Output(
        value=df,
        metadata={
            "row_count": MetadataValue.int(len(df)),
            "preview": MetadataValue.md(_preview_markdown(df)),
            "columns": MetadataValue.json(list(df.columns)),
            "source_file": MetadataValue.text(config.ucc_file),
        }
    )


@asset(
    description="Load master mapping file with SECP_NAME to SECP_NAME_MAPPED",
    compute_kind="pandas",
    group_name="ingestion",
    tags={"domain": "ucc-mapping", "stage": "ingestion"},
)
def master_mapping_data(context: AssetExecutionContext, config: PipelineConfig) -> Output[pd.DataFrame]:
    """Load master mapping file."""
    df = pd.read_csv(
        config.master_file,
        sep=_detect_delimiter(config.master_file),
        encoding='latin-1',
        usecols=RAW_MATCH_COLUMNS + ['SECP_NAME_MAPPED', 'Firm ID'],
        dtype={'Firm ID': 'Int64'},
    ).drop_duplicates()
    df = _standardize_state_columns(df)
    
    df = df[df['SECP_NAME'].notna() & (df['SECP_NAME'].str.strip() != '')]
    df = df.reset_index(drop=True)
    
    context.log.info(f"Loaded {len(df):,} master mapping records")
    
    return Output(
        value=df,
        metadata={
            "row_count": MetadataValue.int(len(df)),
            "preview": MetadataValue.md(_preview_markdown(df)),
            "source_file": MetadataValue.text(config.master_file),
        }
    )


@asset(
    description="Precompute normalized master data for faster matching",
    compute_kind="python",
    group_name="ingestion",
    tags={"domain": "ucc-mapping", "stage": "ingestion"},
)
def normalized_master_data(
    context: AssetExecutionContext, 
    master_mapping_data: pd.DataFrame
) -> dict:
    """Precompute normalized master lookups."""
    master_df = master_mapping_data.copy()
    master_df['SECP_NAME_NORMALIZED'] = master_df['SECP_NAME'].apply(normalize_name)
    master_df = master_df[master_df['SECP_NAME_NORMALIZED'].str.len() > 0]
    master_df['_RAW_STATE_KEY'] = _state_name_keys(master_df, 'SECP_NAME')
    master_df['_NORM_STATE_KEY'] = _state_name_keys(master_df, 'SECP_NAME_NORMALIZED', normalized=False)
    master_df['_STATE_PAIR_KEY'] = _state_pair_keys(master_df)

    raw_candidates = (
        master_df.groupby('_RAW_STATE_KEY')['SECP_NAME_MAPPED']
        .apply(_unique_candidates)
        .to_dict()
    )
    norm_candidates = (
        master_df.groupby('_NORM_STATE_KEY')['SECP_NAME_MAPPED']
        .apply(_unique_candidates)
        .to_dict()
    )
    ambiguous_raw_keys = {key for key, values in raw_candidates.items() if len(values) > 1}
    ambiguous_norm_keys = {key for key, values in norm_candidates.items() if len(values) > 1}

    # Keep first occurrence for each normalized name within its debtor/secured-party state pair.
    raw_dedup = master_df.drop_duplicates('_RAW_STATE_KEY', keep='first')
    master_dedup = master_df.drop_duplicates('_NORM_STATE_KEY', keep='first').reset_index(drop=True)
    
    # Pre-compute entity types for all master names (avoids recomputation in matching)
    normalized_list = master_dedup['SECP_NAME_NORMALIZED'].tolist()
    entity_types = [detect_entity_type(name) for name in normalized_list]
    state_pair_list = master_dedup['_STATE_PAIR_KEY'].tolist()
    state_pair_to_indices = {}
    for idx, state_pair in enumerate(state_pair_list):
        state_pair_to_indices.setdefault(state_pair, []).append(idx)
    
    # Create lookups
    result = {
        'raw_key_to_mapped': dict(zip(raw_dedup['_RAW_STATE_KEY'], raw_dedup['SECP_NAME_MAPPED'])),
        'raw_key_to_raw': dict(zip(raw_dedup['_RAW_STATE_KEY'], raw_dedup['SECP_NAME'])),
        'raw_key_to_firm_id': dict(zip(raw_dedup['_RAW_STATE_KEY'], raw_dedup['Firm ID'])),
        'raw_key_to_mapped_candidates': raw_candidates,
        'ambiguous_raw_keys': ambiguous_raw_keys,
        'norm_to_mapped': dict(zip(master_dedup['_NORM_STATE_KEY'], master_dedup['SECP_NAME_MAPPED'])),
        'norm_to_raw': dict(zip(master_dedup['_NORM_STATE_KEY'], master_dedup['SECP_NAME'])),
        'norm_to_firm_id': dict(zip(master_dedup['_NORM_STATE_KEY'], master_dedup['Firm ID'])),
        'normalized_list': normalized_list,
        'raw_list': master_dedup['SECP_NAME'].tolist(),
        'mapped_list': master_dedup['SECP_NAME_MAPPED'].tolist(),
        'firm_id_list': master_dedup['Firm ID'].tolist(),
        'state_pair_list': state_pair_list,
        'state_pair_to_indices': state_pair_to_indices,
        'entity_types': entity_types,  # Pre-computed entity types
        'ambiguous_norms': ambiguous_norm_keys,
        'norm_to_mapped_candidates': norm_candidates,
    }
    
    context.log.info(f"Prepared {len(result['normalized_list']):,} normalized master records")
    context.log.info(f"Found {len(ambiguous_raw_keys):,} ambiguous raw state/name keys")
    context.log.info(f"Found {len(ambiguous_norm_keys):,} ambiguous normalized state/name keys")
    context.log.info(f"Pre-computed entity types for all master records")
    
    return result


@asset(
    description="Pre-compute master embeddings (CPU OPTIMIZED)",
    compute_kind="ml",
    group_name="preprocessing",
    tags={"domain": "ucc-mapping", "stage": "preprocessing"},
)
def master_embeddings_cache(
        context: AssetExecutionContext,
        config: PipelineConfig,
        normalized_master_data: dict,
) -> dict:
    """Pre-compute master embeddings (CPU OPTIMIZED)."""
    from sentence_transformers import SentenceTransformer
    import faiss
    import os
    from concurrent.futures import ThreadPoolExecutor

    # Optimize CPU usage
    if config.cpu_threads > 0:
        os.environ['OMP_NUM_THREADS'] = str(config.cpu_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(config.cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(config.cpu_threads)

    master_normalized_list = normalized_master_data['normalized_list']

    if not master_normalized_list:
        context.log.info("No master data to embed")
        return {'embeddings': None, 'faiss_index': None, 'ngram_index': None}

    context.log.info("Loading embedding model for CPU...")
    try:
        model = SentenceTransformer(config.embedding_model, device='cpu')
    except Exception:
        model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')

    # CPU-optimized encoding
    context.log.info(f"Encoding {len(master_normalized_list):,} master mappings (CPU)...")
    master_embeddings = model.encode(
        master_normalized_list,
        batch_size=config.embed_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        device='cpu',
        convert_to_numpy=True
    )
    master_embeddings = np.ascontiguousarray(master_embeddings, dtype="float32")

    # Optimized FAISS index for CPU
    context.log.info("Building CPU-optimized FAISS index...")
    faiss.omp_set_num_threads(config.cpu_threads if config.cpu_threads > 0 else -1)

    # For CPU, stick with IndexFlatIP for most cases (it's faster than HNSW on CPU for moderate sizes)
    if len(master_normalized_list) > 100000:
        # Only use HNSW for very large datasets
        faiss_index = faiss.IndexHNSWFlat(master_embeddings.shape[1], 16)  # Smaller M for CPU
        faiss_index.hnsw.efSearch = 64  # Reduced for speed
        faiss_index.hnsw.efConstruction = 100
    else:
        faiss_index = faiss.IndexFlatIP(master_embeddings.shape[1])

    faiss_index.add(master_embeddings)

    # Parallel auxiliary computations
    context.log.info("Computing auxiliary data in parallel...")

    def compute_ngram_batch(args):
        start, end, names = args
        local_ngrams = {}
        for i in range(start, end):
            local_ngrams[i] = generate_ngrams(names[i], 3)
        return local_ngrams

    def compute_soundex_batch(names):
        return [soundex(name) for name in names]

    # Build n-gram index in parallel
    ngram_index = NGramIndex(n=3)
    chunk_size = max(1000, len(master_normalized_list) // config.embedding_workers)
    chunks = [
        (i, min(i + chunk_size, len(master_normalized_list)), master_normalized_list)
        for i in range(0, len(master_normalized_list), chunk_size)
    ]

    with ThreadPoolExecutor(max_workers=config.embedding_workers) as executor:
        # Compute n-grams in parallel
        ngram_results = executor.map(compute_ngram_batch, chunks)
        for ngram_dict in ngram_results:
            for idx, ngrams in ngram_dict.items():
                ngram_index.add_precomputed(idx, master_normalized_list[idx], ngrams)

        # Compute soundex in parallel
        master_soundex = compute_soundex_batch(master_normalized_list)

    # Build TF-IDF weighted n-gram index for better n-gram similarity
    context.log.info("Building TF-IDF n-gram index...")
    tfidf_index = TFIDFNGramIndex(n=3)
    for i, name in enumerate(master_normalized_list):
        tfidf_index.add(i, name)
    tfidf_index.freeze()
    context.log.info(f"  TF-IDF index: {len(tfidf_index._idf):,} unique n-grams weighted")

    context.log.info(f"Master embeddings cache ready: {master_embeddings.shape}")

    return {
        'embeddings': master_embeddings,
        'faiss_index': faiss_index,
        'ngram_index': ngram_index,
        'tfidf_idf_weights': tfidf_index._idf,  # pass IDF weights for compute_match_score
        'soundex': master_soundex,
        'model': model,
    }


@asset(
    description="Stage 1: Exact match on DEBT_STATE, SECP_STATE, and raw SECP_NAME",
    compute_kind="python",
    group_name="matching",
    tags={"domain": "ucc-mapping", "stage": "matching"},
)
def exact_match_raw_results(
    context: AssetExecutionContext,
    ucc_raw_data: pd.DataFrame,
    normalized_master_data: dict,
) -> pd.DataFrame:
    """Stage 1: Exact match on raw SECP_NAME within the same state pair."""
    df = ucc_raw_data.copy()
    raw_keys = _state_name_keys(df, 'SECP_NAME')
    master_lookup = normalized_master_data['raw_key_to_mapped']
    raw_to_master = normalized_master_data['raw_key_to_raw']
    raw_to_firm_id = normalized_master_data['raw_key_to_firm_id']
    ambiguous_raw_keys = normalized_master_data['ambiguous_raw_keys']
    raw_candidates = normalized_master_data['raw_key_to_mapped_candidates']
    
    df['SECP_NAME_MAPPED'] = raw_keys.map(master_lookup)
    df['matched_master_raw'] = raw_keys.map(raw_to_master)
    df['Firm_ID'] = raw_keys.map(raw_to_firm_id)
    df['match_method'] = np.where(df['SECP_NAME_MAPPED'].notna(), 'Exact Match (Raw)', None)
    df['match_score'] = np.where(df['SECP_NAME_MAPPED'].notna(), 1.0, np.nan)
    df['confidence'] = np.where(df['SECP_NAME_MAPPED'].notna(), 'AUTO', None)
    df['ambiguous_master_candidates'] = None

    ambiguous_mask = raw_keys.isin(ambiguous_raw_keys) & df['SECP_NAME_MAPPED'].notna()
    if ambiguous_mask.any():
        df.loc[ambiguous_mask, 'match_method'] = 'Exact Match (Raw - State Ambiguous)'
        df.loc[ambiguous_mask, 'confidence'] = 'REVIEW_HIGH'
        df.loc[ambiguous_mask, 'ambiguous_master_candidates'] = raw_keys[ambiguous_mask].map(
            lambda key: "; ".join(raw_candidates.get(key, []))
        )
    
    matched = df['SECP_NAME_MAPPED'].notna().sum()
    ambiguous_count = int(ambiguous_mask.sum())
    context.log.info(
        f"Stage 1 - Exact Match (Raw + states): "
        f"{matched - ambiguous_count:,} auto + {ambiguous_count:,} ambiguous"
    )
    
    return df


@asset(
    description="Stage 2: Exact match on normalized names",
    compute_kind="python",
    group_name="matching",
    tags={"domain": "ucc-mapping", "stage": "matching"},
)
def exact_match_normalized_results(
    context: AssetExecutionContext,
    exact_match_raw_results: pd.DataFrame,
    normalized_master_data: dict,
) -> pd.DataFrame:
    """Stage 2: Exact match on normalized names."""
    df = exact_match_raw_results.copy()
    
    unmatched_mask = df['SECP_NAME_MAPPED'].isna()
    if not unmatched_mask.any():
        context.log.info("Stage 2 - Exact Match (Normalized): 0 matches (none to process)")
        df['SECP_NAME_NORMALIZED'] = df['SECP_NAME'].apply(normalize_name)
        return df
    
    df['SECP_NAME_NORMALIZED'] = df['SECP_NAME'].apply(normalize_name)
    
    norm_keys = _state_name_keys(df, 'SECP_NAME_NORMALIZED', normalized=False)
    norm_to_mapped = normalized_master_data['norm_to_mapped']
    norm_to_raw = normalized_master_data['norm_to_raw']
    norm_to_firm_id = normalized_master_data['norm_to_firm_id']
    ambiguous_norms = normalized_master_data['ambiguous_norms']
    norm_to_mapped_candidates = normalized_master_data['norm_to_mapped_candidates']
    state_pair_to_indices = normalized_master_data['state_pair_to_indices']
    
    matches = norm_keys[unmatched_mask].map(norm_to_mapped)
    matched_mask = unmatched_mask & matches.notna().reindex(df.index, fill_value=False)
    
    df.loc[matched_mask, 'SECP_NAME_MAPPED'] = norm_keys[matched_mask].map(norm_to_mapped)
    df.loc[matched_mask, 'matched_master_raw'] = norm_keys[matched_mask].map(norm_to_raw)
    df.loc[matched_mask, 'Firm_ID'] = norm_keys[matched_mask].map(norm_to_firm_id)
    df.loc[matched_mask, 'match_score'] = 1.0
    
    matched_norm_keys = norm_keys[matched_mask]
    ambiguous_mask = matched_norm_keys.isin(ambiguous_norms)
    matched_indices = matched_norm_keys.index
    
    df.loc[matched_indices, 'match_method'] = 'Exact Match (Normalized)'
    df.loc[matched_indices, 'confidence'] = 'AUTO'
    
    if ambiguous_mask.any():
        amb_idx = matched_norm_keys[ambiguous_mask].index
        df.loc[amb_idx, 'match_method'] = 'Exact Match (Normalized - Ambiguous)'
        df.loc[amb_idx, 'confidence'] = 'REVIEW_HIGH'
        df.loc[amb_idx, 'ambiguous_master_candidates'] = [
            "; ".join(norm_to_mapped_candidates.get(norm_keys.at[i], []))
            for i in amb_idx
        ]
    
    matched = matched_mask.sum()
    ambiguous_count = ambiguous_mask.sum()
    context.log.info(f"Stage 2 - Exact Match (Normalized): {matched - ambiguous_count:,} auto + {ambiguous_count:,} ambiguous")

    remaining_unmatched = df['SECP_NAME_MAPPED'].isna()
    state_pairs = _state_pair_keys(df)
    no_state_pair_mask = remaining_unmatched & ~state_pairs.isin(state_pair_to_indices.keys())
    if no_state_pair_mask.any():
        df.loc[no_state_pair_mask, 'match_method'] = 'No Master State Pair'
        df.loc[no_state_pair_mask, 'confidence'] = NO_MASTER_STATE_PAIR
        context.log.info(
            f"Stage 2 - Skipped {int(no_state_pair_mask.sum()):,} records "
            "with state pairs absent from master_list"
        )
    
    return df


@asset(
    description="Stage 3: Fuzzy matching with enhanced false positive prevention",
    compute_kind="python",
    group_name="matching",
    tags={"domain": "ucc-mapping", "stage": "matching"},
)
def fuzzy_match_results(
        context: AssetExecutionContext,
        config: PipelineConfig,
        exact_match_normalized_results: pd.DataFrame,
        normalized_master_data: dict,
) -> pd.DataFrame:
    """Stage 3: Fuzzy matching with enhanced false positive prevention."""
    from rapidfuzz import process, fuzz

    df = exact_match_normalized_results.copy()

    unmatched_mask = _pending_match_mask(df)
    if not unmatched_mask.any():
        context.log.info("Stage 3 - Fuzzy Match: 0 matches (none to process)")
        return df

    master_normalized_list = normalized_master_data['normalized_list']
    master_entity_types = normalized_master_data['entity_types']
    norm_to_mapped = normalized_master_data['norm_to_mapped']
    norm_to_raw = normalized_master_data['norm_to_raw']
    norm_to_firm_id = normalized_master_data['norm_to_firm_id']
    ambiguous_norms = normalized_master_data['ambiguous_norms']
    norm_to_mapped_candidates = normalized_master_data['norm_to_mapped_candidates']
    state_pair_to_indices = normalized_master_data['state_pair_to_indices']

    pending = df.loc[unmatched_mask, STATE_COLUMNS + ['SECP_NAME_NORMALIZED']].copy()
    pending['_match_key'] = _state_name_keys(pending, 'SECP_NAME_NORMALIZED', normalized=False)
    key_to_indices = pending.groupby('_match_key').apply(lambda g: g.index.tolist()).to_dict()
    unique_pending = pending.drop_duplicates('_match_key', keep='first')

    queries = unique_pending['SECP_NAME_NORMALIZED'].tolist()
    indices = unique_pending.index.tolist()
    query_keys = unique_pending['_match_key'].tolist()
    query_state_pairs = _state_pair_keys(unique_pending).tolist()

    deduped_count = int(unmatched_mask.sum()) - len(queries)
    if deduped_count:
        context.log.info(
            f"Stage 3 - Fuzzy Match: deduped {deduped_count:,} repeated normalized state/name records"
        )

    # Pre-compute entity types for all queries
    query_entity_types = [detect_entity_type(q) for q in queries]

    auto_count = 0
    review_count = 0
    rejected_count = 0  # Track rejections for debugging

    batch_size = max(1, config.fuzzy_batch_size)
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        batch_queries = queries[start:end]
        batch_indices = indices[start:end]
        batch_keys = query_keys[start:end]
        batch_query_types = query_entity_types[start:end]

        batch_state_pairs = query_state_pairs[start:end]

        positions_by_state_pair = {}
        for pos, state_pair in enumerate(batch_state_pairs):
            positions_by_state_pair.setdefault(state_pair, []).append(pos)

        for state_pair, positions in positions_by_state_pair.items():
            candidate_indices = state_pair_to_indices.get(state_pair, [])
            if not candidate_indices:
                continue

            candidate_names = [master_normalized_list[candidate_idx] for candidate_idx in candidate_indices]
            candidate_types = [master_entity_types[candidate_idx] for candidate_idx in candidate_indices]
            sub_queries = [batch_queries[pos] for pos in positions]

            scores = process.cdist(
                sub_queries,
                candidate_names,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=config.fuzzy_review_threshold,
                workers=config.fuzzy_workers
            )

            if scores.size == 0 or np.max(scores) < config.fuzzy_review_threshold:
                continue

            best_idx = scores.argmax(axis=1)
            best_scores = scores[np.arange(len(sub_queries)), best_idx]

            for sub_i, pos in enumerate(positions):
                idx = batch_indices[pos]
                update_indices = key_to_indices[batch_keys[pos]]
                score = float(best_scores[sub_i])
                if score < config.fuzzy_review_threshold:
                    continue

                match_idx = int(candidate_indices[int(best_idx[sub_i])])
                match_norm = master_normalized_list[match_idx]
                query_name = batch_queries[pos]

                # Entity type gating
                query_type = batch_query_types[pos]
                match_type = candidate_types[int(best_idx[sub_i])]
                if not entity_types_compatible(query_type, match_type):
                    rejected_count += 1
                    continue

                # ENHANCED FALSE POSITIVE PREVENTION
                # Check for meaningful identifier overlap
                if not has_shared_meaningful_identifier(query_name, match_norm):
                    context.log.debug(f"Rejected: '{query_name}' -> '{match_norm}' (no shared meaningful identifiers)")
                    rejected_count += 1
                    continue

                # Use enhanced combined fuzzy score
                combined_score = combined_fuzzy_score(query_name, match_norm)

                # CRITICAL: Minimum combined score to prevent false positives
                # combined_fuzzy_score returns 0-100 scale; require at least 85
                FUZZY_COMBINED_MIN = 85.0
                if combined_score < FUZZY_COMBINED_MIN:
                    context.log.debug(
                        f"Rejected: '{query_name}' -> '{match_norm}' (combined score {combined_score:.1f} < {FUZZY_COMBINED_MIN})")
                    rejected_count += 1
                    continue

                # Additional check for entity core mismatch
                query_core = extract_entity_core(query_name)
                match_core = extract_entity_core(match_norm)

                if query_core != match_core and len(query_core) > 2 and len(match_core) > 2:
                    # Check if cores are similar (allow for minor variations)
                    core_similarity = fuzz.ratio(query_core, match_core)
                    if core_similarity < 70:
                        # Only allow if it's a clear acronym match
                        if not (is_acronym_match(query_core, match_core) or is_acronym_match(match_core, query_core)):
                            context.log.debug(
                                f"Rejected: '{query_name}' -> '{match_norm}' (core mismatch: '{query_core}' vs '{match_core}')")
                            rejected_count += 1
                            continue

                final_score = combined_score
                match_key = (state_pair[0], state_pair[1], match_norm)
                is_ambiguous = match_key in ambiguous_norms

                df.loc[update_indices, 'SECP_NAME_MAPPED'] = norm_to_mapped.get(match_key)
                df.loc[update_indices, 'matched_master_raw'] = norm_to_raw.get(match_key)
                df.loc[update_indices, 'Firm_ID'] = norm_to_firm_id.get(match_key)
                df.loc[update_indices, 'match_score'] = final_score / 100.0

                if is_ambiguous:
                    df.loc[update_indices, 'match_method'] = 'Fuzzy Match (Ambiguous)'
                    df.loc[update_indices, 'confidence'] = 'REVIEW_HIGH'

                    candidates_raw = norm_to_mapped_candidates.get(match_key, [])
                    if candidates_raw:
                        valid_candidates = []
                        for candidate in candidates_raw:
                            if candidate is not None:
                                if isinstance(candidate, float) and pd.isna(candidate):
                                    continue
                                valid_candidates.append(str(candidate))

                        df.loc[update_indices, 'ambiguous_master_candidates'] = "; ".join(valid_candidates)
                    else:
                        df.loc[update_indices, 'ambiguous_master_candidates'] = ""

                    review_count += len(update_indices)
                elif final_score >= config.fuzzy_auto_threshold:
                    df.loc[update_indices, 'match_method'] = 'Fuzzy Match'
                    df.loc[update_indices, 'confidence'] = 'AUTO'
                    auto_count += len(update_indices)
                else:
                    df.loc[update_indices, 'match_method'] = 'Fuzzy Match'
                    df.loc[update_indices, 'confidence'] = 'REVIEW_HIGH'
                    review_count += len(update_indices)

    context.log.info(
        f"Stage 3 - Fuzzy Match: {auto_count:,} auto + {review_count:,} review (rejected {rejected_count:,} false positives)")

    return df


@asset(
    description="Stage 4: Advanced multi-signal embedding match",
    compute_kind="ml",
    group_name="matching",
    tags={"domain": "ucc-mapping", "stage": "matching"},
)
def advanced_embedding_results(
    context: AssetExecutionContext,
    config: PipelineConfig,
    fuzzy_match_results: pd.DataFrame,
    normalized_master_data: dict,
    master_embeddings_cache: dict,
) -> pd.DataFrame:
    """Stage 4: Advanced multi-signal matching."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    # Optimize CPU usage
    if config.cpu_threads > 0:
        os.environ['OMP_NUM_THREADS'] = str(config.cpu_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(config.cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(config.cpu_threads)

    df = fuzzy_match_results.copy()
    
    unmatched_mask = _pending_match_mask(df)
    if not unmatched_mask.any():
        context.log.info("Stage 4 - Advanced Match: 0 matches (none to process)")
        return df
    
    master_normalized_list = normalized_master_data['normalized_list']
    master_mapped_list = normalized_master_data['mapped_list']
    master_raw_list = normalized_master_data['raw_list']
    master_firm_id_list = normalized_master_data['firm_id_list']
    master_entity_types = normalized_master_data['entity_types']
    master_state_pair_list = normalized_master_data['state_pair_list']
    state_pair_to_indices = normalized_master_data['state_pair_to_indices']
    
    if not master_normalized_list:
        context.log.info("Stage 4 - Advanced Match: 0 matches (no master data)")
        return df

    # Use pre-computed embeddings and indexes from cache (NO duplicate computation)
    master_embeddings = master_embeddings_cache.get('embeddings')
    faiss_index = master_embeddings_cache.get('faiss_index')
    ngram_index = master_embeddings_cache.get('ngram_index')
    master_soundex = master_embeddings_cache.get('soundex')
    model = master_embeddings_cache.get('model')
    tfidf_idf_weights = master_embeddings_cache.get('tfidf_idf_weights')  # TF-IDF n-gram weights
    
    # Fallback: If model failed to deserialize, reload it
    if model is None:
        from sentence_transformers import SentenceTransformer
        context.log.warning("Model not found in cache, reloading...")
        try:
            model = SentenceTransformer(config.embedding_model, device='cpu')
        except Exception:
            model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')

    # Fallback: If embeddings/index failed to deserialize, rebuild them
    if master_embeddings is None:
        context.log.error("Critical: Master embeddings cache is invalid!")
        return df

    if ngram_index is None:
        ngram_index = NGramIndex(n=3)
        for i, name in enumerate(master_normalized_list):
            ngram_index.add(i, name)

    # Fallback: Rebuild soundex if needed
    if master_soundex is None:
        master_soundex = [soundex(name) for name in master_normalized_list]

    pending = df.loc[unmatched_mask, STATE_COLUMNS + ['SECP_NAME_NORMALIZED']].copy()
    pending['_match_key'] = _state_name_keys(pending, 'SECP_NAME_NORMALIZED', normalized=False)
    key_to_indices = pending.groupby('_match_key').apply(lambda g: g.index.tolist()).to_dict()
    unique_pending = pending.drop_duplicates('_match_key', keep='first')

    queries = unique_pending['SECP_NAME_NORMALIZED'].tolist()
    indices = unique_pending.index.tolist()
    query_keys = unique_pending['_match_key'].tolist()
    query_state_pairs = _state_pair_keys(unique_pending).tolist()

    deduped_count = int(unmatched_mask.sum()) - len(queries)
    if deduped_count:
        context.log.info(
            f"Stage 4 - Advanced Match: deduped {deduped_count:,} repeated normalized state/name records"
        )

    context.log.info(f"Matching {len(queries):,} records with CPU-optimized processing...")

    # Pre-compute all query metadata in parallel
    context.log.info("Pre-computing query metadata in parallel...")

    def compute_query_metadata(query):
        return {
            'soundex': soundex(query),
            'entity_type': detect_entity_type(query),
            'ngram_candidates': set(ngram_index.query(query, min_overlap=2))
        }

    with ThreadPoolExecutor(max_workers=config.embedding_workers) as executor:
        query_metadata = list(executor.map(compute_query_metadata, queries))

    auto_count = high_count = medium_count = 0
    batch_size = config.embed_batch_size
    batch_size = config.embed_batch_size
    top_k = max(1, min(config.embed_top_k, len(master_normalized_list)))

    # Process in smaller batches for CPU efficiency
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        batch_queries = queries[start:end]
        batch_indices = indices[start:end]
        batch_keys = query_keys[start:end]
        batch_metadata = query_metadata[start:end]
        batch_state_pairs = query_state_pairs[start:end]

        # Encode batch (CPU optimized)
        query_embeddings = model.encode(
            batch_queries,
            batch_size=128,  # Smaller batch for CPU
            normalize_embeddings=True,
            show_progress_bar=False,
            device='cpu'
        )
        query_embeddings = np.ascontiguousarray(query_embeddings, dtype="float32")

        positions_by_state_pair = {}
        for pos, state_pair in enumerate(batch_state_pairs):
            positions_by_state_pair.setdefault(state_pair, []).append(pos)

        for state_pair, positions in positions_by_state_pair.items():
            candidate_indices = state_pair_to_indices.get(state_pair, [])
            if not candidate_indices:
                continue

            candidate_array = np.asarray(candidate_indices, dtype=np.int64)
            candidate_embeddings = master_embeddings[candidate_array]
            state_query_embeddings = query_embeddings[positions]
            similarity_matrix = np.matmul(state_query_embeddings, candidate_embeddings.T)
            group_top_k = min(top_k, len(candidate_array))

            for row_pos, pos in enumerate(positions):
                query_name = batch_queries[pos]
                idx = batch_indices[pos]
                update_indices = key_to_indices[batch_keys[pos]]
                metadata = batch_metadata[pos]
                query_soundex = metadata['soundex']
                query_type = metadata['entity_type']
                ngram_candidates = metadata['ngram_candidates']

                sims = similarity_matrix[row_pos]
                if len(candidate_array) > group_top_k:
                    top_local = np.argpartition(sims, -group_top_k)[-group_top_k:]
                    top_local = top_local[np.argsort(sims[top_local])[::-1]]
                else:
                    top_local = np.argsort(sims)[::-1]

                top_embed_candidates = candidate_array[top_local].tolist()
                state_candidates = set(candidate_indices)
                all_candidates = set(top_embed_candidates) | (ngram_candidates & state_candidates)
                if len(all_candidates) > config.max_candidates_per_query:
                    remaining_slots = config.max_candidates_per_query - len(top_embed_candidates)
                    ngram_only = list((ngram_candidates & state_candidates) - set(top_embed_candidates))[:remaining_slots]
                    all_candidates = set(top_embed_candidates + ngram_only)

                embed_sim_map = {
                    int(candidate_array[local_idx]): float(sims[local_idx])
                    for local_idx in top_local
                }
                for cand_idx in all_candidates:
                    if cand_idx not in embed_sim_map:
                        embed_sim_map[cand_idx] = float(
                            np.dot(query_embeddings[pos], master_embeddings[cand_idx])
                        )

                sorted_candidates = sorted(
                    all_candidates,
                    key=lambda x: embed_sim_map[x],
                    reverse=True
                )

                best_score = 0.0
                best_match_idx = -1

                # Process candidates with early termination
                for cand_idx in sorted_candidates:
                    embed_sim = embed_sim_map[cand_idx]

                    # Early termination conditions
                    if embed_sim < 0.70:
                        break  # Remaining candidates will be even worse

                    if best_score >= config.early_stop_threshold:
                        break  # Found excellent match, stop searching

                    master_name = master_normalized_list[cand_idx]
                    master_type = master_entity_types[cand_idx]
                    if master_state_pair_list[cand_idx] != state_pair:
                        continue

                    # Fast pre-filters
                    if not entity_types_compatible(query_type, master_type):
                        continue

                    if embed_sim < 0.85 and not min_token_overlap(query_name, master_name, min_overlap=1):
                        continue

                    # Compute multi-signal score (optimized, with TF-IDF n-grams)
                    total_score, _ = compute_match_score(
                        query_name, master_name, embed_sim,
                        phonetic_weight=0.08,  # Slightly reduced for speed
                        ngram_weight=0.12,
                        fuzzy_weight=0.20,
                        embed_weight=0.50,  # Increased embedding weight
                        acronym_weight=0.10,
                        idf_weights=tfidf_idf_weights,
                    )

                    # Quick bonuses
                    if query_soundex == master_soundex[cand_idx]:
                        total_score += 0.05
                    if query_type == master_type and query_type != 'OTHER':
                        total_score += 0.03

                    if total_score > best_score:
                        best_score = total_score
                        best_match_idx = cand_idx

                # Apply results
                if best_match_idx >= 0 and best_score >= config.embed_medium_threshold:
                    df.loc[update_indices, 'SECP_NAME_MAPPED'] = master_mapped_list[best_match_idx]
                    df.loc[update_indices, 'matched_master_raw'] = master_raw_list[best_match_idx]
                    df.loc[update_indices, 'Firm_ID'] = master_firm_id_list[best_match_idx]
                    df.loc[update_indices, 'match_score'] = best_score

                    if best_score >= config.embed_auto_threshold:
                        df.loc[update_indices, 'match_method'] = 'Advanced Match'
                        df.loc[update_indices, 'confidence'] = 'AUTO'
                        auto_count += len(update_indices)
                    elif best_score >= config.embed_high_threshold:
                        df.loc[update_indices, 'match_method'] = 'Advanced Match'
                        df.loc[update_indices, 'confidence'] = 'REVIEW_HIGH'
                        high_count += len(update_indices)
                    else:
                        df.loc[update_indices, 'match_method'] = 'Advanced Match'
                        df.loc[update_indices, 'confidence'] = 'REVIEW_MEDIUM'
                        medium_count += len(update_indices)

        # Progress and memory management
        if start > 0 and start % 500 == 0:
            gc.collect()
            context.log.info(f"    Processed {start + len(batch_queries):,} / {len(queries):,}")

    context.log.info(f"Stage 4 - Advanced Match: {auto_count:,} auto, {high_count:,} high, {medium_count:,} medium")

    df.attrs['embedding_model'] = model
    gc.collect()

    return df


@asset(
    description="Stage 5: Cluster remaining unmatched records",
    compute_kind="ml",
    group_name="clustering",
    tags={"domain": "ucc-mapping", "stage": "clustering"},
)
def clustered_results(
    context: AssetExecutionContext,
    config: PipelineConfig,
    advanced_embedding_results: pd.DataFrame,
    master_embeddings_cache: dict,
) -> pd.DataFrame:
    """Stage 5: Cluster remaining unmatched records."""
    from rapidfuzz import fuzz
    import faiss
    
    df = advanced_embedding_results.copy()
    
    unmatched_mask = _pending_match_mask(df)
    
    if not unmatched_mask.any():
        context.log.info("Stage 5 - Clustering: 0 records to cluster")
        df['cluster_id'] = None
        df['cluster_label'] = None
        df['cluster_size'] = None
        return df
    
    sub_df = df.loc[unmatched_mask, ['SECP_NAME_NORMALIZED']].copy()
    names = sub_df['SECP_NAME_NORMALIZED'].tolist()
    original_indices = sub_df.index.tolist()
    n = len(names)
    
    context.log.info(f"Stage 5 - Clustering {n:,} unmatched records...")
    
    # NOTE: OMP/MKL thread env vars are set at module level (top of file)
    # Setting them here would have no effect since numpy/FAISS are already loaded.
    
    # Use model from cache (no reloading)
    model = master_embeddings_cache.get('model')
    if model is None:
        from sentence_transformers import SentenceTransformer
        try:
            model = SentenceTransformer(config.embedding_model, device='cpu')
        except Exception:
            model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
    else:
        context.log.info("Using model from cache (no reload)")
    
    context.log.info("Encoding unmatched names...")
    embed_dim = int(getattr(model, "get_sentence_embedding_dimension", lambda: 384)() or 384)
    embeddings = np.empty((n, embed_dim), dtype="float32")
    
    # Use larger batch size for better throughput
    encode_bs = 256
    for start in range(0, n, encode_bs):
        end = min(n, start + encode_bs)
        batch = names[start:end]
        batch_embeddings = model.encode(
            batch,
            batch_size=encode_bs,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        embeddings[start:end] = np.asarray(batch_embeddings, dtype="float32")
        
        # Progress logging for large datasets
        if start > 0 and start % 5000 == 0:
            context.log.info(f"  Encoded {start:,} / {n:,} names")
    
    context.log.info("Building search index...")
    dim = embeddings.shape[1]
    if n >= config.cluster_hnsw_threshold:
        index = faiss.IndexHNSWFlat(dim, config.cluster_hnsw_m)
        index.hnsw.efSearch = config.cluster_hnsw_ef_search
        index.hnsw.efConstruction = config.cluster_hnsw_ef_construction
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    context.log.info("Finding clusters...")
    uf = UnionFind(n)
    k = min(config.cluster_k_neighbors, n)
    
    search_batch_size = min(2048, config.search_batch_size)
    
    for start in range(0, n, search_batch_size):
        end = min(n, start + search_batch_size)
        query_batch = embeddings[start:end]
        
        scores, indices_batch = index.search(query_batch, k=k)
        
        for qi in range(end - start):
            i = start + qi
            for j in range(1, k):
                neighbor_idx = int(indices_batch[qi, j])
                sim = float(scores[qi, j])
                
                if sim >= config.cluster_sim_threshold:
                    fuzzy_score = fuzz.token_sort_ratio(names[i], names[neighbor_idx])
                    if fuzzy_score >= config.cluster_fuzzy_guard:
                        uf.union(i, neighbor_idx)
        
        if start > 0 and start % 10000 == 0:
            gc.collect()
    
    context.log.info("Assigning cluster labels...")
    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int32)
    unique_roots, cluster_ids = np.unique(roots, return_inverse=True)
    
    cluster_id_str = [f"C{cid:06d}" for cid in cluster_ids]
    cluster_sizes = pd.Series(cluster_id_str).value_counts().to_dict()
    cluster_size_arr = [cluster_sizes[cid] for cid in cluster_id_str]
    
    cluster_labels = {}
    cluster_to_indices = {}
    for i, cid in enumerate(cluster_id_str):
        cluster_to_indices.setdefault(cid, []).append(i)
    
    for cid, idx_list in cluster_to_indices.items():
        if len(idx_list) == 1:
            cluster_labels[cid] = names[idx_list[0]]
        elif len(idx_list) > 50:
            cluster_labels[cid] = max((names[i] for i in idx_list), key=len)
        else:
            cluster_embeddings = np.asarray(embeddings[idx_list], dtype=np.float64)
            centroid = cluster_embeddings.mean(axis=0, keepdims=True)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            medoid_idx = int(np.argmin(distances))
            cluster_labels[cid] = names[idx_list[medoid_idx]]
    
    cluster_label_arr = [cluster_labels[cid] for cid in cluster_id_str]
    
    df['cluster_id'] = None
    df['cluster_label'] = None
    df['cluster_size'] = None
    
    df.loc[original_indices, 'cluster_id'] = cluster_id_str
    df.loc[original_indices, 'cluster_label'] = cluster_label_arr
    df.loc[original_indices, 'cluster_size'] = cluster_size_arr
    df.loc[original_indices, 'match_method'] = 'Unmatched (Clustered)'
    df.loc[original_indices, 'confidence'] = 'NEW_ENTITY'
    
    n_clusters = len(unique_roots)
    singleton_clusters = sum(1 for s in cluster_sizes.values() if s == 1)
    context.log.info(f"Stage 5 - Clustering: {n_clusters:,} clusters ({singleton_clusters:,} singletons)")
    
    df.attrs['embedding_model'] = model
    df.attrs['cluster_embeddings'] = embeddings
    df.attrs['cluster_names'] = names
    df.attrs['cluster_indices'] = original_indices
    
    del index
    gc.collect()
    
    return df


@asset(
    description="Stage 6: Hierarchical merging of clusters",
    compute_kind="ml",
    group_name="clustering",
    tags={"domain": "ucc-mapping", "stage": "clustering"},
)
def merged_cluster_results(
    context: AssetExecutionContext,
    config: PipelineConfig,
    clustered_results: pd.DataFrame,
    master_embeddings_cache: dict,
) -> pd.DataFrame:
    """Stage 6: Hierarchical merging of clusters."""
    import faiss
    
    df = clustered_results.copy()
    
    if 'cluster_id' not in df.columns:
        return df
    
    clustered_mask = df['cluster_id'].notna()
    if not clustered_mask.any():
        return df
    
    # Use model from cache (no reloading)
    model = master_embeddings_cache.get('model')
    if model is None:
        from sentence_transformers import SentenceTransformer
        try:
            model = SentenceTransformer(config.embedding_model, device='cpu')
        except Exception:
            model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
    else:
        context.log.info("Using model from cache (no reload)")
    
    n_iterations = config.cluster_merge_iterations
    merge_k = config.cluster_merge_k
    max_cluster_size = config.cluster_max_size
    
    for iteration in range(n_iterations):
        cluster_info = df.loc[clustered_mask, ['cluster_id', 'cluster_label']].drop_duplicates('cluster_id')
        cluster_sizes = df.loc[clustered_mask, 'cluster_id'].value_counts().to_dict()
        
        if len(cluster_info) <= 1:
            break
        
        n_clusters = len(cluster_info)
        
        if n_clusters > 100000:
            context.log.info(f"Iteration {iteration + 1}: Skipping ({n_clusters:,} clusters exceeds 100k limit)")
            break
        
        context.log.info(f"Iteration {iteration + 1}: Processing {n_clusters:,} clusters (max size cap: {max_cluster_size})...")
        
        labels = cluster_info['cluster_label'].tolist()
        cluster_ids = cluster_info['cluster_id'].tolist()
        
        # Track cluster sizes for merge decisions
        cluster_idx_to_size = {i: cluster_sizes.get(cid, 1) for i, cid in enumerate(cluster_ids)}
        
        label_embeddings = model.encode(
            labels,
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        label_embeddings = np.asarray(label_embeddings, dtype="float32")
        
        n = len(labels)
        uf = UnionFind(n)
        
        # Track merged cluster sizes to prevent mega-clusters
        merged_sizes = {i: cluster_idx_to_size[i] for i in range(n)}
        
        index = faiss.IndexFlatIP(label_embeddings.shape[1])
        index.add(label_embeddings)
        
        # Use strict threshold - NO relaxation on later iterations
        base_threshold = config.cluster_merge_threshold
        
        k = min(merge_k, n)
        scores, indices = index.search(label_embeddings, k=k)
        
        merges_made = 0
        merges_blocked_by_size = 0
        for i in range(n):
            for j_idx in range(1, k):
                j = int(indices[i, j_idx])
                sim = float(scores[i, j_idx])
                
                if should_merge_clusters(labels[i], labels[j], sim):
                    root_i = uf.find(i)
                    root_j = uf.find(j)
                    if root_i != root_j:
                        # Check if combined size would exceed max
                        combined_size = merged_sizes[root_i] + merged_sizes[root_j]
                        if combined_size <= max_cluster_size:
                            uf.union(i, j)
                            # Update size tracking
                            new_root = uf.find(i)
                            merged_sizes[new_root] = combined_size
                            merges_made += 1
                            # Log merge decisions for debugging (first 10 merges)
                            if merges_made <= 10:
                                from rapidfuzz import fuzz
                                fuzzy_score = fuzz.token_set_ratio(labels[i].lower(), labels[j].lower())
                                context.log.debug(
                                    f"  Merging: '{labels[i][:40]}...' <-> '{labels[j][:40]}...' "
                                    f"(embed={sim:.3f}, fuzzy={fuzzy_score})"
                                )
                        else:
                            merges_blocked_by_size += 1
        
        if merges_made == 0:
            if merges_blocked_by_size > 0:
                context.log.info(f"No merges made ({merges_blocked_by_size:,} blocked by size cap), stopping")
            else:
                context.log.info("No merges possible, stopping")
            del index, label_embeddings
            gc.collect()
            break
        
        if merges_blocked_by_size > 0:
            context.log.info(f"  {merges_blocked_by_size:,} potential merges blocked by size cap")
        
        roots = [uf.find(i) for i in range(n)]
        unique_roots, new_cluster_ids = np.unique(roots, return_inverse=True)
        
        old_to_new = {}
        new_label_candidates = {}
        
        for i, old_cid in enumerate(cluster_ids):
            new_cid = f"H{iteration}{new_cluster_ids[i]:06d}"
            old_to_new[old_cid] = new_cid
            new_label_candidates.setdefault(new_cid, []).append(labels[i])
        
        # Pick best label for merged cluster
        # Strategy: Prefer concise, complete names over very long ones
        new_labels = {}
        for new_cid, candidates in new_label_candidates.items():
            if len(candidates) == 1:
                new_labels[new_cid] = candidates[0]
            else:
                # Filter out very long names (likely truncated or less representative)
                reasonable_length = [c for c in candidates if len(c) <= 100]
                if reasonable_length:
                    candidates = reasonable_length
                
                # Prefer multi-word names (more descriptive)
                multi_word = [c for c in candidates if ' ' in c.strip()]
                if multi_word:
                    # Among multi-word, prefer moderate length (not too short, not too long)
                    # This avoids picking very long names like "UNITED STATES OF AMERICA ACTING THROUGH RURAL"
                    # when a shorter, more representative name exists
                    def score_label(label):
                        word_count = len(label.split())
                        length = len(label)
                        # Prefer 2-5 word names of moderate length
                        if 2 <= word_count <= 5 and 20 <= length <= 80:
                            return (0, -word_count, length)  # Best: moderate word count, moderate length
                        elif word_count >= 2:
                            return (1, -word_count, length)  # Good: multi-word
                        else:
                            return (2, length)  # Less preferred: single word
                    
                    new_labels[new_cid] = min(multi_word, key=score_label)
                else:
                    # Single-word candidates - pick shortest (most concise)
                    new_labels[new_cid] = min(candidates, key=len)
        
        df.loc[clustered_mask, 'cluster_id'] = df.loc[clustered_mask, 'cluster_id'].map(old_to_new)
        df.loc[clustered_mask, 'cluster_label'] = df.loc[clustered_mask, 'cluster_id'].map(new_labels)
        
        n_merged = len(unique_roots)
        context.log.info(f"{n_clusters:,} -> {n_merged:,} clusters (merged {n_clusters - n_merged:,})")
        
        del index, label_embeddings
        gc.collect()
        
        if n_merged < 100 or (n_clusters - n_merged) < 50:
            break
    
    new_sizes = df.loc[clustered_mask, 'cluster_id'].value_counts().to_dict()
    df.loc[clustered_mask, 'cluster_size'] = df.loc[clustered_mask, 'cluster_id'].map(new_sizes)
    
    final_clusters = df.loc[clustered_mask, 'cluster_id'].nunique()
    context.log.info(f"Stage 6 - Final: {final_clusters:,} clusters")
    
    return df


@asset(
    description="Stage 7: Suggest mappings for clusters",
    compute_kind="ml",
    group_name="clustering",
    tags={"domain": "ucc-mapping", "stage": "clustering"},
)
def suggested_mapping_results(
    context: AssetExecutionContext,
    config: PipelineConfig,
    merged_cluster_results: pd.DataFrame,
    normalized_master_data: dict,
    master_embeddings_cache: dict,
) -> pd.DataFrame:
    """Stage 7: Suggest mappings for clusters."""
    from rapidfuzz import fuzz
    import faiss

    df = merged_cluster_results.copy()

    if 'cluster_id' not in df.columns:
        return df

    # CRITICAL: Only suggest mappings for unmatched clusters (NEW_ENTITY)
    clustered_mask = (df['cluster_id'].notna()) & (df['confidence'] == 'NEW_ENTITY')
    if not clustered_mask.any():
        context.log.info("Stage 7 - No unmatched clusters to suggest mappings for")
        return df

    # Use pre-computed master data
    master_mapped = normalized_master_data['mapped_list']
    master_mapped_normalized = normalized_master_data['normalized_list']
    master_state_pairs = normalized_master_data['state_pair_list']

    if not master_mapped:
        context.log.info("Stage 7 - No master mappings to suggest from")
        return df

    cluster_info = df.loc[
        clustered_mask,
        ['cluster_id', 'cluster_label', 'DEBT_STATE', 'SECP_STATE']
    ].drop_duplicates('cluster_id')
    cluster_labels = cluster_info['cluster_label'].tolist()
    cluster_ids = cluster_info['cluster_id'].tolist()
    cluster_state_pairs = _state_pair_keys(cluster_info).tolist()

    context.log.info(f"Stage 7 - Suggesting mappings for {len(cluster_ids):,} clusters...")
    context.log.info(f"  Master mappings available: {len(master_mapped):,}")

    # Use pre-computed master embeddings and index
    master_embeddings = master_embeddings_cache.get('embeddings')
    index = master_embeddings_cache.get('faiss_index')
    model = master_embeddings_cache.get('model')

    # Fallback handling
    if model is None:
        from sentence_transformers import SentenceTransformer
        context.log.warning("Model not found in cache, reloading...")
        try:
            model = SentenceTransformer(config.embedding_model, device='cpu')
        except Exception:
            model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')

    if master_embeddings is None or index is None:
        import faiss
        context.log.warning("Embeddings/index not found in cache, rebuilding...")
        master_embeddings = model.encode(
            master_mapped_normalized,
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        master_embeddings = np.ascontiguousarray(master_embeddings, dtype="float32")
        faiss.omp_set_num_threads(1)
        index = faiss.IndexFlatIP(master_embeddings.shape[1])
        index.add(master_embeddings)

    context.log.info("Using pre-computed master embeddings")

    # Pre-normalize all labels
    context.log.info("Normalizing cluster labels...")
    label_norms = [normalize_name(l) for l in cluster_labels]

    # Filter out empty normalized labels
    valid_indices = [i for i, norm in enumerate(label_norms) if norm.strip()]
    if not valid_indices:
        context.log.warning("No valid normalized cluster labels")
        df['suggested_mapping'] = None
        df['suggestion_score'] = None
        df['suggestion_confidence'] = None
        df['alternative_suggestions'] = None
        return df

    # Filter to valid labels only
    valid_cluster_ids = [cluster_ids[i] for i in valid_indices]
    valid_cluster_labels = [cluster_labels[i] for i in valid_indices]
    valid_label_norms = [label_norms[i] for i in valid_indices]
    valid_state_pairs = [cluster_state_pairs[i] for i in valid_indices]

    context.log.info(f"Processing {len(valid_cluster_ids):,} valid clusters")

    # Encode cluster labels
    context.log.info("Encoding cluster labels...")
    label_embeddings = model.encode(
        valid_label_norms,
        batch_size=512,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    label_embeddings = np.asarray(label_embeddings, dtype="float32")
    context.log.info(f"  Encoded {len(label_embeddings):,} cluster labels")

    k = min(max(3, config.embed_top_k), len(master_mapped))
    scores, indices = index.search(label_embeddings, k=k)
    context.log.info(f"  Searched for top-{k} matches for each cluster")

    SUGGEST_HIGH_THRESHOLD = 0.80
    SUGGEST_MEDIUM_THRESHOLD = 0.70
    SUGGEST_LOW_THRESHOLD = 0.60

    # Build results dictionaries
    cluster_suggestions = {}
    suggestion_scores = {}
    suggestion_confidence = {}
    suggestion_alternatives = {}
    no_suggest_scores = []

    for i, cid in enumerate(valid_cluster_ids):
        state_pair = valid_state_pairs[i]
        scored_candidates = []
        for rank in range(indices.shape[1]):
            candidate_idx = int(indices[i, rank])
            if candidate_idx >= len(master_mapped) or master_state_pairs[candidate_idx] != state_pair:
                continue

            label_norm = valid_label_norms[i]
            match_norm = master_mapped_normalized[candidate_idx]
            fuzzy_score = fuzz.token_set_ratio(label_norm.lower(), match_norm.lower()) / 100.0
            score = (float(scores[i, rank]) * 0.6) + (fuzzy_score * 0.4)
            if label_norm.lower() in match_norm.lower() or match_norm.lower() in label_norm.lower():
                score += 0.1
            scored_candidates.append((score, candidate_idx, float(scores[i, rank])))

        if not scored_candidates:
            continue

        scored_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        score, best_idx, _ = scored_candidates[0]

        # Determine confidence level
        if score >= SUGGEST_HIGH_THRESHOLD:
            confidence = 'SUGGEST_HIGH'
        elif score >= SUGGEST_MEDIUM_THRESHOLD:
            confidence = 'SUGGEST_MEDIUM'
        elif score >= SUGGEST_LOW_THRESHOLD:
            confidence = 'SUGGEST_LOW'
        else:
            confidence = None

        if confidence is not None:
            best_match = master_mapped[best_idx]

            cluster_suggestions[cid] = best_match
            suggestion_scores[cid] = score
            suggestion_confidence[cid] = confidence

            # Get alternatives with proper validation
            alts = []
            for _, alt_idx, alt_embed_score in scored_candidates[1:]:
                if alt_embed_score >= SUGGEST_LOW_THRESHOLD * 0.9 and alt_idx < len(master_mapped):
                    alt_suggestion = master_mapped[alt_idx]

                    # Validate the alternative suggestion
                    if alt_suggestion is not None:
                        if isinstance(alt_suggestion, float) and pd.isna(alt_suggestion):
                            continue  # Skip NaN values

                        # Convert to string and validate
                        alt_str = str(alt_suggestion).strip()
                        if alt_str and alt_str.lower() != 'nan':
                            alts.append(alt_str)

            # Join alternatives safely
            suggestion_alternatives[cid] = "; ".join(alts) if alts else None
        else:
            no_suggest_scores.append(score)

    # Initialize columns
    df['suggested_mapping'] = None
    df['suggestion_score'] = None
    df['suggestion_confidence'] = None
    df['alternative_suggestions'] = None

    # Vectorized assignment
    if cluster_suggestions:
        cluster_to_suggestion = pd.Series(cluster_suggestions)
        cluster_to_score = pd.Series(suggestion_scores)
        cluster_to_confidence = pd.Series(suggestion_confidence)
        cluster_to_alternatives = pd.Series(suggestion_alternatives)

        mask = df['cluster_id'].isin(cluster_to_suggestion.index)
        df.loc[mask, 'suggested_mapping'] = df.loc[mask, 'cluster_id'].map(cluster_to_suggestion)
        df.loc[mask, 'suggestion_score'] = df.loc[mask, 'cluster_id'].map(cluster_to_score)
        df.loc[mask, 'suggestion_confidence'] = df.loc[mask, 'cluster_id'].map(cluster_to_confidence)
        df.loc[mask, 'alternative_suggestions'] = df.loc[mask, 'cluster_id'].map(cluster_to_alternatives)

    # Summary statistics
    high_count = sum(1 for v in suggestion_confidence.values() if v == 'SUGGEST_HIGH')
    medium_count = sum(1 for v in suggestion_confidence.values() if v == 'SUGGEST_MEDIUM')
    low_count = sum(1 for v in suggestion_confidence.values() if v == 'SUGGEST_LOW')
    no_suggest = len(valid_cluster_ids) - len(cluster_suggestions)

    # Diagnostic logging
    if no_suggest_scores:
        no_suggest_scores = np.array(no_suggest_scores)
        context.log.info(
            f"  Score distribution for unmatched clusters: "
            f"min={no_suggest_scores.min():.3f}, max={no_suggest_scores.max():.3f}, "
            f"mean={no_suggest_scores.mean():.3f}, median={np.median(no_suggest_scores):.3f}"
        )

    context.log.info(
        f"Stage 7 - Suggestions: {high_count:,} high, {medium_count:,} medium, {low_count:,} low, {no_suggest:,} no match")

    # Cleanup
    del label_embeddings
    gc.collect()

    return df


@asset(
    description="Load tracked banks file mapping bank names to Firm_ID",
    compute_kind="pandas",
    group_name="ingestion",
    tags={"domain": "ucc-mapping", "stage": "ingestion"},
)
def our_universe_data(
    context: AssetExecutionContext,
    config: PipelineConfig,
) -> Output[pd.DataFrame]:
    """Load tracked banks and normalize them to the pipeline lookup schema."""
    if not Path(config.universe_file).exists():
        context.log.warning(
            f"Tracked banks file not found at {config.universe_file}; "
            "continuing with Firm_ID values from the master mapping file only"
        )
        df = pd.DataFrame(columns=["SECP_NAME", "Firm_ID"])
        return Output(
            value=df,
            metadata={
                "row_count": MetadataValue.int(0),
                "unique_firm_ids": MetadataValue.int(0),
                "preview": MetadataValue.md("No tracked banks file found."),
                "source_file": MetadataValue.text(config.universe_file),
            },
        )

    df = pd.read_csv(
        config.universe_file,
        sep=_detect_delimiter(config.universe_file),
        encoding="latin-1",
        usecols=["Bank Name", "Firm ID"],
        dtype={"Firm ID": "Int64"},
    ).drop_duplicates()
    df = df.rename(columns={"Bank Name": "SECP_NAME", "Firm ID": "Firm_ID"})

    df = df[df["SECP_NAME"].notna() & (df["SECP_NAME"].str.strip() != "")]
    df = df.reset_index(drop=True)

    context.log.info(
        f"Loaded {len(df):,} tracked bank rows | "
        f"{df['Firm_ID'].notna().sum():,} with Firm_ID"
    )

    return Output(
        value=df,
        metadata={
            "row_count": MetadataValue.int(len(df)),
            "unique_firm_ids": MetadataValue.int(int(df["Firm_ID"].nunique())),
            "preview": MetadataValue.md(_preview_markdown(df)),
            "source_file": MetadataValue.text(config.universe_file),
        },
    )


@asset(
    description="Enrich pipeline results with Firm_ID from tracked banks",
    compute_kind="pandas",
    group_name="reporting",
    tags={"domain": "ucc-mapping", "stage": "reporting"},
)
def universe_enriched_results(
    context: AssetExecutionContext,
    suggested_mapping_results: pd.DataFrame,
    our_universe_data: pd.DataFrame,
) -> pd.DataFrame:
    """Join tracked-bank Firm_ID values onto the matched results.

    Matching logic (applied in priority order):
      1. Exact match on SECP_NAME_MAPPED → tracked Bank Name → Firm_ID.
      2. Exact match on raw SECP_NAME → tracked Bank Name → Firm_ID, only where
         the mapped-name lookup did not produce a Firm_ID.
    """
    df = suggested_mapping_results.copy()
    universe = our_universe_data.copy()
    existing_firm_id = (
        df["Firm_ID"].copy()
        if "Firm_ID" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="object")
    )
    if universe.empty:
        df["Firm_ID"] = existing_firm_id
        df["GU_COMPANY_NAME_universe"] = pd.NA
        context.log.info(
            f"Tracked-bank enrichment skipped: {int(df['Firm_ID'].notna().sum()):,} "
            "rows keep Firm_ID from master mapping"
        )
        return df

    # ------------------------------------------------------------------
    # Build case-insensitive lookup tables from the tracked banks file.
    # All keys are stored upper-cased so lookups are case-insensitive.
    # When multiple tracked-bank rows map to the same upper-cased key we keep
    # the one whose Firm_ID appears most frequently (majority vote), which
    # prevents a rare mis-keyed row from overriding the correct Firm_ID.
    # ------------------------------------------------------------------

    def _majority_firm(sub: pd.DataFrame) -> pd.Series:
        """Return the most common Firm_ID for a tracked bank name."""
        if sub["Firm_ID"].notna().any():
            firm = sub["Firm_ID"].dropna().mode().iloc[0]
        else:
            firm = pd.NA
        return pd.Series({"Firm_ID": firm})

    universe["_key"] = universe["SECP_NAME"].str.upper().str.strip()
    univ_by_secp = (
        universe[universe["_key"].notna()]
        .groupby("_key", as_index=False)
        .apply(_majority_firm, include_groups=False)
    )
    secp_to_firm_ci = univ_by_secp.set_index("_key")["Firm_ID"]

    # ------------------------------------------------------------------
    # Initialise output columns
    # ------------------------------------------------------------------
    df["Firm_ID"] = existing_firm_id
    df["GU_COMPANY_NAME_universe"] = pd.NA

    # ------------------------------------------------------------------
    # Pass 1 – SECP_NAME_MAPPED is the authoritative resolved identity.
    # Always resolve Firm_ID from the mapped name first so that, e.g.,
    # "FIRST SECURITY BANK" mapped to "Glacier Bank" gets Firm_ID 5959
    # (Glacier Bancorp) rather than 1354 (First Security Bancorp).
    # ------------------------------------------------------------------
    if "SECP_NAME_MAPPED" in df.columns:
        mapped_upper = df["SECP_NAME_MAPPED"].str.upper().str.strip()

        mapped_firm_secp = mapped_upper.map(secp_to_firm_ci)

        has_mapped_firm = mapped_firm_secp.notna()

        df.loc[has_mapped_firm, "Firm_ID"] = mapped_firm_secp[has_mapped_firm]

        pass1_firm = int(has_mapped_firm.sum())
        context.log.info(
            f"Tracked-bank join - Pass 1 (SECP_NAME_MAPPED, case-insensitive): "
            f"{pass1_firm:,} Firm_IDs matched"
        )

    # ------------------------------------------------------------------
    # Pass 2 – fall back to raw SECP_NAME only where SECP_NAME_MAPPED
    # produced no result (i.e. unmatched / new-entity rows).
    # ------------------------------------------------------------------
    secp_upper = df["SECP_NAME"].str.upper().str.strip()

    raw_firm = secp_upper.map(secp_to_firm_ci)

    fill_raw_firm = df["Firm_ID"].isna() & raw_firm.notna()

    df.loc[fill_raw_firm, "Firm_ID"] = raw_firm[fill_raw_firm]

    pass2_firm = int(fill_raw_firm.sum())
    context.log.info(
        f"Tracked-bank join - Pass 2 (raw SECP_NAME fallback, case-insensitive): "
        f"{pass2_firm:,} additional Firm_IDs"
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_with_firm = int(df["Firm_ID"].notna().sum())
    total = len(df)
    context.log.info(
        f"Tracked-bank enrichment complete: "
        f"{total_with_firm:,}/{total:,} rows have Firm_ID "
        f"({total_with_firm / total * 100:.1f}%)"
    )

    return df


@asset(
    description="Generate output files for different confidence levels",
    compute_kind="io",
    group_name="reporting",
    tags={"domain": "ucc-mapping", "stage": "reporting"},
)
def pipeline_outputs(
    context: AssetExecutionContext,
    config: PipelineConfig,
    universe_enriched_results: pd.DataFrame,
) -> MaterializeResult:
    """Generate output files for different confidence levels."""
    df = universe_enriched_results.copy()
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_cols = ['DEBT_STATE', 'SECP_STATE', 'SECP_NAME', 'SECP_NAME_NORMALIZED']
    match_cols = [
        'SECP_NAME_MAPPED',
        'matched_master_raw',
        'match_method',
        'match_score',
        'confidence',
        'ambiguous_master_candidates'
    ]
    cluster_cols = ['cluster_id', 'cluster_label', 'cluster_size']
    suggestion_cols = ['suggested_mapping', 'suggestion_score', 'suggestion_confidence', 'alternative_suggestions']
    universe_cols = ['Firm_ID', 'GU_COMPANY_NAME_universe']
    
    all_cols = base_cols + match_cols + cluster_cols + suggestion_cols + universe_cols
    available_cols = [c for c in all_cols if c in df.columns]
    
    output_df = df[available_cols].copy()
    
    # Auto-mapped
    auto_mask = output_df['confidence'] == 'AUTO'
    auto_df = output_df[auto_mask].sort_values(['match_method', 'SECP_NAME'])
    auto_path = output_dir / f"auto_mapped_{timestamp}.csv"
    auto_df.to_csv(auto_path, index=False)
    context.log.info(f"Auto-mapped: {len(auto_df):,} records -> {auto_path}")
    
    # Review needed
    review_mask = output_df['confidence'].isin(['REVIEW_HIGH', 'REVIEW_MEDIUM', 'REVIEW_LOW'])
    review_df = output_df[review_mask].sort_values(['confidence', 'match_score'], ascending=[True, False])
    review_path = output_dir / f"review_needed_{timestamp}.csv"
    review_df.to_csv(review_path, index=False)
    context.log.info(f"Review needed: {len(review_df):,} records -> {review_path}")
    
    # New entities
    new_mask = output_df['confidence'] == 'NEW_ENTITY'
    new_df = output_df[new_mask].sort_values(
        ['suggestion_confidence', 'cluster_size', 'cluster_id', 'SECP_NAME'],
        ascending=[True, False, True, True],
        na_position='last'
    )
    new_path = output_dir / f"new_entities_{timestamp}.csv"
    new_df.to_csv(new_path, index=False)
    context.log.info(f"New entities: {len(new_df):,} records -> {new_path}")
    
    # Full results
    full_path = output_dir / f"full_results_{timestamp}.csv"
    output_df.to_csv(full_path, index=False)
    context.log.info(f"Full results: {len(output_df):,} records -> {full_path}")

    # ------------------------------------------------------------------
    # Exact, Fuzzy, and Advanced master matches — slim export (deduped on
    # DEBT_STATE + SECP_STATE + SECP_NAME) and left-join onto the full raw UCC file. Unmatched rows
    # are excluded.
    # ------------------------------------------------------------------
    def _master_match_mask(series: pd.Series) -> pd.Series:
        s = series.astype("string")
        return series.notna() & (
            s.str.startswith("Exact Match", na=False)
            | s.str.startswith("Fuzzy Match", na=False)
            | s.str.startswith("Advanced Match", na=False)
        )

    enrich_cols = [
        "SECP_NAME_MAPPED",
        "match_method",
        "match_score",
        "confidence",
        "Firm_ID",
        "GU_COMPANY_NAME",
    ]
    ef_mask = _master_match_mask(df["match_method"])
    ef_source = df.loc[ef_mask].copy()
    if "GU_COMPANY_NAME_universe" in ef_source.columns:
        ef_source["GU_COMPANY_NAME"] = ef_source["GU_COMPANY_NAME_universe"]
    else:
        ef_source["GU_COMPANY_NAME"] = pd.NA

    n_before_dedup = len(ef_source)
    join_key_cols = [col for col in RAW_MATCH_COLUMNS if col in ef_source.columns]
    ef_dedup = ef_source.drop_duplicates(subset=join_key_cols, keep="first")
    n_dup_dropped = n_before_dedup - len(ef_dedup)
    if n_dup_dropped:
        context.log.info(
            f"Slim / raw-join lookup: dropped {n_dup_dropped:,} duplicate state/name "
            f"rows (keep=first) before export and merge"
        )

    slim_cols = [
        "DEBT_STATE",
        "SECP_STATE",
        "SECP_NAME",
        "SECP_NAME_MAPPED",
        "match_method",
        "match_score",
        "confidence",
        "Firm_ID",
        "GU_COMPANY_NAME",
    ]
    slim_available = [c for c in slim_cols if c in ef_dedup.columns]
    ef_slim = ef_dedup[slim_available].copy()
    ef_slim_path = output_dir / f"exact_fuzzy_mapped_{timestamp}.csv"
    ef_slim.to_csv(ef_slim_path, index=False)
    context.log.info(
        f"Master-matched slim (deduped state/name): {len(ef_slim):,} records -> {ef_slim_path}"
    )

    raw_mapped_path = None
    raw_mapped_rows = 0
    raw_with_mapping = 0

    if not all(col in ef_dedup.columns for col in RAW_MATCH_COLUMNS):
        context.log.warning(
            "Enriched dataframe missing state/name join columns; skipping raw merge"
        )
    else:
        map_for_join = ef_dedup[RAW_MATCH_COLUMNS + enrich_cols].drop_duplicates(
            subset=RAW_MATCH_COLUMNS,
            keep="first",
        )
        try:
            raw_full = pd.read_csv(
                config.ucc_file,
                sep=_detect_delimiter(config.ucc_file),
                encoding="latin-1",
            )
        except Exception as e:
            context.log.error(f"Could not read full raw UCC for merge: {e}")
        else:
            if not all(col in raw_full.columns for col in RAW_MATCH_COLUMNS):
                context.log.warning(
                    "Raw file missing state/name join columns; skipping raw merge"
                )
            else:
                raw_full = _standardize_state_columns(raw_full)
                raw_enriched = raw_full.merge(
                    map_for_join,
                    on=RAW_MATCH_COLUMNS,
                    how="left",
                )
                raw_mapped_path = output_dir / f"raw_mapped_with_firm_ids_{timestamp}.csv"
                raw_enriched.to_csv(raw_mapped_path, index=False)
                raw_mapped_rows = len(raw_enriched)
                raw_with_mapping = int(
                    raw_enriched["SECP_NAME_MAPPED"].notna().sum()
                )
                context.log.info(
                    f"Raw mapped output (left join on DEBT_STATE + SECP_STATE + SECP_NAME): "
                    f"{raw_mapped_rows:,} rows, "
                    f"{raw_with_mapping:,} with SECP_NAME_MAPPED and Firm_ID "
                    f"-> {raw_mapped_path}"
                )
    
    # Summary
    summary_path = output_dir / f"summary_{timestamp}.txt"
    method_counts = output_df['match_method'].value_counts().to_dict()
    confidence_counts = output_df['confidence'].value_counts().to_dict()
    
    with open(summary_path, 'w') as f:
        f.write("UCC Mapping Pipeline Summary (Dagster)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Overall Results:\n")
        f.write(f"  Total UCC Records: {len(df):,}\n")
        f.write(f"  Auto-Mapped: {len(auto_df):,}\n")
        f.write(f"  Review Needed: {len(review_df):,}\n")
        f.write(f"  New Entities: {len(new_df):,}\n")
        
        f.write("\nBy Match Method:\n")
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {method}: {count:,}\n")
        
        f.write("\nBy Confidence Level:\n")
        for conf, count in sorted(confidence_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {conf}: {count:,}\n")
        
        if new_mask.any():
            n_clusters = output_df.loc[new_mask, 'cluster_id'].nunique()
            avg_size = output_df.loc[new_mask, 'cluster_size'].mean()
            f.write(f"\nClustering Stats:\n")
            f.write(f"  Number of clusters: {n_clusters:,}\n")
            f.write(f"  Average cluster size: {avg_size:.1f}\n")

        # Universe coverage
        if 'Firm_ID' in output_df.columns:
            with_firm = output_df['Firm_ID'].notna().sum()
            f.write(f"\nTracked Bank Coverage:\n")
            f.write(f"  Records with Firm_ID:         {with_firm:,}\n")

        f.write("\nExact / Fuzzy / Advanced (excludes Unmatched); slim deduped on state/name:\n")
        f.write(f"  Slim export rows: {len(ef_slim):,}\n")
        if raw_mapped_path is not None:
            f.write(
                f"  Raw mapped-with-firm-ids file rows: {raw_mapped_rows:,} "
                f"({raw_with_mapping:,} rows with SECP_NAME_MAPPED filled)\n"
            )

    firm_matched = int(output_df['Firm_ID'].notna().sum()) if 'Firm_ID' in output_df.columns else 0
    auto_pct = len(auto_df) / len(df) * 100 if len(df) > 0 else 0
    
    meta = {
        "total_records": MetadataValue.int(len(df)),
        "auto_mapped_count": MetadataValue.int(len(auto_df)),
        "auto_mapped_pct": MetadataValue.float(auto_pct),
        "review_needed_count": MetadataValue.int(len(review_df)),
        "new_entities_count": MetadataValue.int(len(new_df)),
        "firm_id_matched_count": MetadataValue.int(firm_matched),
        "mapping_slim_deduped_count": MetadataValue.int(len(ef_slim)),
        "output_dir": MetadataValue.path(str(output_dir)),
        "timestamp": MetadataValue.text(timestamp),
    }
    if raw_mapped_path is not None:
        meta["raw_mapped_with_firm_ids_rows"] = MetadataValue.int(raw_mapped_rows)
        meta["raw_mapped_with_firm_ids_with_mapping"] = MetadataValue.int(
            raw_with_mapping
        )
        meta["raw_mapped_with_firm_ids_path"] = MetadataValue.path(str(raw_mapped_path))
    return MaterializeResult(metadata=meta)

