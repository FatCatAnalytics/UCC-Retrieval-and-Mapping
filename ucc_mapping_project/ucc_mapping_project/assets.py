
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
        sep=",",
        encoding="latin-1",
        usecols=['SECP_NAME', 'SECP_STATE']
    ).drop_duplicates()
    
    df = df[df['SECP_NAME'].notna() & (df['SECP_NAME'].str.strip() != '')]
    df = df.reset_index(drop=True)
    
    context.log.info(f"Loaded {len(df):,} unique UCC records")
    
    # Return Output with metadata facets
    return Output(
        value=df,
        metadata={
            "row_count": MetadataValue.int(len(df)),
            "preview": MetadataValue.md(df.head().to_markdown()),
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
        sep=",",
        encoding='utf-8',
        usecols=['SECP_NAME', 'SECP_NAME_MAPPED']
    ).drop_duplicates()
    
    df = df[df['SECP_NAME'].notna() & (df['SECP_NAME'].str.strip() != '')]
    df = df.reset_index(drop=True)
    
    context.log.info(f"Loaded {len(df):,} master mapping records")
    
    return Output(
        value=df,
        metadata={
            "row_count": MetadataValue.int(len(df)),
            "preview": MetadataValue.md(df.head().to_markdown()),
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
    
    # Identify ambiguous normalized keys
    norm_counts = master_df['SECP_NAME_NORMALIZED'].value_counts()
    ambiguous_norms = set(norm_counts[norm_counts > 1].index.tolist())
    
    # Keep first occurrence for each normalized name
    master_dedup = master_df.drop_duplicates('SECP_NAME_NORMALIZED', keep='first')
    
    # Pre-compute entity types for all master names (avoids recomputation in matching)
    normalized_list = master_dedup['SECP_NAME_NORMALIZED'].tolist()
    entity_types = [detect_entity_type(name) for name in normalized_list]
    
    # Create lookups
    result = {
        'norm_to_mapped': dict(zip(master_dedup['SECP_NAME_NORMALIZED'], master_dedup['SECP_NAME_MAPPED'])),
        'norm_to_raw': dict(zip(master_dedup['SECP_NAME_NORMALIZED'], master_dedup['SECP_NAME'])),
        'normalized_list': normalized_list,
        'raw_list': master_dedup['SECP_NAME'].tolist(),
        'mapped_list': master_dedup['SECP_NAME_MAPPED'].tolist(),
        'entity_types': entity_types,  # Pre-computed entity types
        'ambiguous_norms': ambiguous_norms,
        'norm_to_mapped_candidates': master_df.groupby('SECP_NAME_NORMALIZED')['SECP_NAME_MAPPED'].apply(list).to_dict(),
        'raw_lookup': master_mapping_data.set_index('SECP_NAME')['SECP_NAME_MAPPED'].to_dict(),
    }
    
    context.log.info(f"Prepared {len(result['normalized_list']):,} normalized master records")
    context.log.info(f"Found {len(ambiguous_norms):,} ambiguous normalized keys")
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
    description="Stage 1: Exact match on raw SECP_NAME",
    compute_kind="python",
    group_name="matching",
    tags={"domain": "ucc-mapping", "stage": "matching"},
)
def exact_match_raw_results(
    context: AssetExecutionContext,
    ucc_raw_data: pd.DataFrame,
    normalized_master_data: dict,
) -> pd.DataFrame:
    """Stage 1: Exact match on raw SECP_NAME."""
    df = ucc_raw_data.copy()
    master_lookup = normalized_master_data['raw_lookup']
    
    df['SECP_NAME_MAPPED'] = df['SECP_NAME'].map(master_lookup)
    df['match_method'] = np.where(df['SECP_NAME_MAPPED'].notna(), 'Exact Match (Raw)', None)
    df['match_score'] = np.where(df['SECP_NAME_MAPPED'].notna(), 1.0, np.nan)
    df['confidence'] = np.where(df['SECP_NAME_MAPPED'].notna(), 'AUTO', None)
    
    matched = df['SECP_NAME_MAPPED'].notna().sum()
    context.log.info(f"Stage 1 - Exact Match (Raw): {matched:,} matches")
    
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
    
    norm_to_mapped = normalized_master_data['norm_to_mapped']
    norm_to_raw = normalized_master_data['norm_to_raw']
    ambiguous_norms = normalized_master_data['ambiguous_norms']
    norm_to_mapped_candidates = normalized_master_data['norm_to_mapped_candidates']
    
    matches = df.loc[unmatched_mask, 'SECP_NAME_NORMALIZED'].map(norm_to_mapped)
    matched_mask = unmatched_mask & matches.notna().reindex(df.index, fill_value=False)
    
    df.loc[matched_mask, 'SECP_NAME_MAPPED'] = df.loc[matched_mask, 'SECP_NAME_NORMALIZED'].map(norm_to_mapped)
    df.loc[matched_mask, 'matched_master_raw'] = df.loc[matched_mask, 'SECP_NAME_NORMALIZED'].map(norm_to_raw)
    df.loc[matched_mask, 'match_score'] = 1.0
    
    matched_norms = df.loc[matched_mask, 'SECP_NAME_NORMALIZED']
    ambiguous_mask = matched_norms.isin(ambiguous_norms)
    matched_indices = matched_norms.index
    
    df.loc[matched_indices, 'match_method'] = 'Exact Match (Normalized)'
    df.loc[matched_indices, 'confidence'] = 'AUTO'
    
    if ambiguous_mask.any():
        amb_idx = matched_norms[ambiguous_mask].index
        df.loc[amb_idx, 'match_method'] = 'Exact Match (Normalized - Ambiguous)'
        df.loc[amb_idx, 'confidence'] = 'REVIEW_HIGH'
        df.loc[amb_idx, 'ambiguous_master_candidates'] = df.loc[
            amb_idx, 'SECP_NAME_NORMALIZED'
        ].map(lambda n: "; ".join(norm_to_mapped_candidates.get(n, [])))
    
    matched = matched_mask.sum()
    ambiguous_count = ambiguous_mask.sum()
    context.log.info(f"Stage 2 - Exact Match (Normalized): {matched - ambiguous_count:,} auto + {ambiguous_count:,} ambiguous")
    
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

    unmatched_mask = df['SECP_NAME_MAPPED'].isna() & (df['SECP_NAME_NORMALIZED'].str.len() > 0)
    if not unmatched_mask.any():
        context.log.info("Stage 3 - Fuzzy Match: 0 matches (none to process)")
        return df

    master_normalized_list = normalized_master_data['normalized_list']
    master_entity_types = normalized_master_data['entity_types']
    norm_to_mapped = normalized_master_data['norm_to_mapped']
    norm_to_raw = normalized_master_data['norm_to_raw']
    ambiguous_norms = normalized_master_data['ambiguous_norms']
    norm_to_mapped_candidates = normalized_master_data['norm_to_mapped_candidates']

    queries = df.loc[unmatched_mask, 'SECP_NAME_NORMALIZED'].tolist()
    indices = df.index[unmatched_mask].tolist()

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
        batch_query_types = query_entity_types[start:end]

        scores = process.cdist(
            batch_queries,
            master_normalized_list,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=config.fuzzy_review_threshold,
            workers=config.fuzzy_workers
        )

        if scores.size == 0 or np.max(scores) < config.fuzzy_review_threshold:
            continue

        best_idx = scores.argmax(axis=1)
        best_scores = scores[np.arange(len(batch_queries)), best_idx]

        for i, idx in enumerate(batch_indices):
            score = float(best_scores[i])
            if score < config.fuzzy_review_threshold:
                continue

            match_idx = int(best_idx[i])
            match_norm = master_normalized_list[match_idx]
            query_name = batch_queries[i]

            # Entity type gating
            query_type = batch_query_types[i]
            match_type = master_entity_types[match_idx]
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
            is_ambiguous = match_norm in ambiguous_norms

            df.at[idx, 'SECP_NAME_MAPPED'] = norm_to_mapped.get(match_norm)
            df.at[idx, 'matched_master_raw'] = norm_to_raw.get(match_norm)
            df.at[idx, 'match_score'] = final_score / 100.0

            if is_ambiguous:
                df.at[idx, 'match_method'] = 'Fuzzy Match (Ambiguous)'
                df.at[idx, 'confidence'] = 'REVIEW_HIGH'

                candidates_raw = norm_to_mapped_candidates.get(match_norm, [])
                if candidates_raw:
                    valid_candidates = []
                    for candidate in candidates_raw:
                        if candidate is not None:
                            if isinstance(candidate, float) and pd.isna(candidate):
                                continue
                            valid_candidates.append(str(candidate))

                    df.at[idx, 'ambiguous_master_candidates'] = "; ".join(valid_candidates)
                else:
                    df.at[idx, 'ambiguous_master_candidates'] = ""

                review_count += 1
            elif final_score >= config.fuzzy_auto_threshold:
                df.at[idx, 'match_method'] = 'Fuzzy Match'
                df.at[idx, 'confidence'] = 'AUTO'
                auto_count += 1
            else:
                df.at[idx, 'match_method'] = 'Fuzzy Match'
                df.at[idx, 'confidence'] = 'REVIEW_HIGH'
                review_count += 1

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
    
    unmatched_mask = df['SECP_NAME_MAPPED'].isna() & (df['SECP_NAME_NORMALIZED'].str.len() > 0)
    if not unmatched_mask.any():
        context.log.info("Stage 4 - Advanced Match: 0 matches (none to process)")
        return df
    
    master_normalized_list = normalized_master_data['normalized_list']
    master_mapped_list = normalized_master_data['mapped_list']
    master_raw_list = normalized_master_data['raw_list']
    master_entity_types = normalized_master_data['entity_types']
    
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
    if master_embeddings is None or faiss_index is None:
        context.log.error("Critical: Master embeddings cache is invalid!")
        return df

    if ngram_index is None:
        ngram_index = NGramIndex(n=3)
        for i, name in enumerate(master_normalized_list):
            ngram_index.add(i, name)

    # Fallback: Rebuild soundex if needed
    if master_soundex is None:
        master_soundex = [soundex(name) for name in master_normalized_list]

    queries = df.loc[unmatched_mask, 'SECP_NAME_NORMALIZED'].tolist()
    indices = df.index[unmatched_mask].tolist()

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
    top_k = min(config.embed_top_k, len(master_normalized_list))

    # Use larger batch size for efficiency
    batch_size = config.embed_batch_size
    top_k = min(config.embed_top_k, len(master_normalized_list))

    # Process in smaller batches for CPU efficiency
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        batch_queries = queries[start:end]
        batch_indices = indices[start:end]
        batch_metadata = query_metadata[start:end]

        # Encode batch (CPU optimized)
        query_embeddings = model.encode(
            batch_queries,
            batch_size=128,  # Smaller batch for CPU
            normalize_embeddings=True,
            show_progress_bar=False,
            device='cpu'
        )
        query_embeddings = np.ascontiguousarray(query_embeddings, dtype="float32")

        # FAISS search with reduced top-k
        top_k_scores, top_k_indices = faiss_index.search(query_embeddings, k=top_k)

        # Process each query with aggressive optimization
        for j, (query_name, idx) in enumerate(zip(batch_queries, batch_indices)):
            metadata = batch_metadata[j]
            query_soundex = metadata['soundex']
            query_type = metadata['entity_type']
            ngram_candidates = metadata['ngram_candidates']

            # Combine candidates with priority to FAISS results
            faiss_candidates = set(top_k_indices[j].tolist())

            # Limit total candidates for speed
            all_candidates = faiss_candidates | ngram_candidates
            if len(all_candidates) > config.max_candidates_per_query:
                # Prioritize FAISS candidates (they're sorted by similarity)
                faiss_list = list(faiss_candidates)
                remaining_slots = config.max_candidates_per_query - len(faiss_list)
                ngram_only = list(ngram_candidates - faiss_candidates)[:remaining_slots]
                all_candidates = set(faiss_list + ngram_only)

            # Pre-compute embedding similarities for all candidates
            embed_sim_map = {}
            for k in range(min(top_k, len(top_k_indices[j]))):
                cand_idx = int(top_k_indices[j][k])
                embed_sim_map[cand_idx] = float(top_k_scores[j][k])

            # For non-FAISS candidates, compute embeddings
            for cand_idx in all_candidates:
                if cand_idx not in embed_sim_map:
                    embed_sim_map[cand_idx] = float(
                        np.dot(query_embeddings[j], master_embeddings[cand_idx])
                    )

            # Sort candidates by embedding similarity for early termination
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
                df.at[idx, 'SECP_NAME_MAPPED'] = master_mapped_list[best_match_idx]
                df.at[idx, 'matched_master_raw'] = master_raw_list[best_match_idx]
                df.at[idx, 'match_score'] = best_score

                if best_score >= config.embed_auto_threshold:
                    df.at[idx, 'match_method'] = 'Advanced Match'
                    df.at[idx, 'confidence'] = 'AUTO'
                    auto_count += 1
                elif best_score >= config.embed_high_threshold:
                    df.at[idx, 'match_method'] = 'Advanced Match'
                    df.at[idx, 'confidence'] = 'REVIEW_HIGH'
                    high_count += 1
                else:
                    df.at[idx, 'match_method'] = 'Advanced Match'
                    df.at[idx, 'confidence'] = 'REVIEW_MEDIUM'
                    medium_count += 1

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
    
    unmatched_mask = (
        df['SECP_NAME_MAPPED'].isna() &
        df['SECP_NAME_NORMALIZED'].notna() &
        (df['SECP_NAME_NORMALIZED'].str.len() > 0)
    )
    
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

    if not master_mapped:
        context.log.info("Stage 7 - No master mappings to suggest from")
        return df

    cluster_info = df.loc[clustered_mask, ['cluster_id', 'cluster_label']].drop_duplicates('cluster_id')
    cluster_labels = cluster_info['cluster_label'].tolist()
    cluster_ids = cluster_info['cluster_id'].tolist()

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

    k = 3
    scores, indices = index.search(label_embeddings, k=k)
    context.log.info(f"  Searched for top-{k} matches for each cluster")

    SUGGEST_HIGH_THRESHOLD = 0.80
    SUGGEST_MEDIUM_THRESHOLD = 0.70
    SUGGEST_LOW_THRESHOLD = 0.60

    # Vectorized processing
    context.log.info("Computing fuzzy scores (vectorized)...")
    best_match_indices = indices[:, 0].astype(int)
    best_match_norms = [master_mapped_normalized[idx] for idx in best_match_indices]

    from rapidfuzz import process
    fuzzy_scores_array = np.array([
        fuzz.token_set_ratio(ln.lower(), mn.lower()) / 100.0
        for ln, mn in zip(valid_label_norms, best_match_norms)
    ])

    # Vectorized scores
    embed_scores = scores[:, 0].astype(float)
    combined_scores = (embed_scores * 0.6) + (fuzzy_scores_array * 0.4)

    # Substring boost
    substring_boost = np.array([
        0.1 if (ln.lower() in mn.lower() or mn.lower() in ln.lower())
        else 0.0
        for ln, mn in zip(valid_label_norms, best_match_norms)
    ])
    combined_scores += substring_boost

    # Build results dictionaries
    cluster_suggestions = {}
    suggestion_scores = {}
    suggestion_confidence = {}
    suggestion_alternatives = {}

    for i, cid in enumerate(valid_cluster_ids):
        score = float(combined_scores[i])

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
            best_idx = int(indices[i, 0])
            best_match = master_mapped[best_idx]

            cluster_suggestions[cid] = best_match
            suggestion_scores[cid] = score
            suggestion_confidence[cid] = confidence

            # Get alternatives with proper validation
            alts = []
            for j in range(1, k):
                if j < indices.shape[1] and j < scores.shape[1]:  # Bounds check
                    alt_idx = int(indices[i, j])
                    alt_score = float(scores[i, j])

                    if alt_score >= SUGGEST_LOW_THRESHOLD * 0.9 and alt_idx < len(master_mapped):
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
    if no_suggest > 0:
        no_suggest_mask = np.array([cid not in cluster_suggestions for cid in valid_cluster_ids])
        if no_suggest_mask.any():
            no_suggest_scores = combined_scores[no_suggest_mask]
            if len(no_suggest_scores) > 0:
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
    description="Generate output files for different confidence levels",
    compute_kind="io",
    group_name="reporting",
    tags={"domain": "ucc-mapping", "stage": "reporting"},
)
def pipeline_outputs(
    context: AssetExecutionContext,
    config: PipelineConfig,
    suggested_mapping_results: pd.DataFrame,
) -> MaterializeResult:
    """Generate output files for different confidence levels."""
    df = suggested_mapping_results.copy()
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_cols = ['SECP_NAME', 'SECP_STATE', 'SECP_NAME_NORMALIZED']
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
    
    all_cols = base_cols + match_cols + cluster_cols + suggestion_cols
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
    
    auto_pct = len(auto_df) / len(df) * 100 if len(df) > 0 else 0
    
    return MaterializeResult(
        metadata={
            "total_records": MetadataValue.int(len(df)),
            "auto_mapped_count": MetadataValue.int(len(auto_df)),
            "auto_mapped_pct": MetadataValue.float(auto_pct),
            "review_needed_count": MetadataValue.int(len(review_df)),
            "new_entities_count": MetadataValue.int(len(new_df)),
            "output_dir": MetadataValue.path(str(output_dir)),
            "timestamp": MetadataValue.text(timestamp),
        }
    )

