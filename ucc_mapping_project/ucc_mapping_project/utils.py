# =============================================================================
# UTILITIES
# =============================================================================
import re
import math
from typing import List, Set, Dict, Tuple, Optional
from difflib import SequenceMatcher
from collections import Counter
import pandas as pd
from rapidfuzz import fuzz
from .constants import (
    LEGAL_SUFFIXES,
    ROLE_RE,
    COMMON_WORDS,
    ABBREVIATIONS,
    ENTITY_TYPE_PATTERNS
)


def normalize_name(name: str) -> str:
    """Normalize SECP_NAME for better matching."""
    if pd.isna(name) or not str(name).strip():
        return ""
    
    name = str(name).upper().strip()
    
    # Strip role qualifiers first (e.g. "as trustee", "as agent")
    m = ROLE_RE.search(name)
    if m:
        name = name[:m.start()].rstrip(" ,;:-")
    
    # Remove legal suffixes (INC, LLC, etc.) — but NOT entity-type words like BANK
    for suffix in LEGAL_SUFFIXES:
        name = re.sub(suffix, '', name)
    
    # Remove common noise words — apply AFTER suffix stripping
    # Process longer phrases first so they match before their sub-words
    for word in COMMON_WORDS:
        name = re.sub(word, ' ', name)
    
    # Expand known abbreviations
    for abbr, full in ABBREVIATIONS.items():
        name = re.sub(abbr, full, name)
    
    # Strip non-alphanumeric (keep spaces), collapse whitespace
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name)
    name = name.strip()
    
    return name

def soundex(name: str) -> str:
    """Generate Soundex code for phonetic matching (ANSI/AIM standard)."""
    name = name.upper()
    name = re.sub(r'[^A-Z]', '', name)
    if not name:
        return ''

    first_letter = name[0]
    code_map = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6',
    }
    # Vowels and H/W/Y are ignored (mapped to '')
    # Adjacent same-code letters are collapsed

    prev_code = code_map.get(first_letter, '')
    coded = first_letter
    for char in name[1:]:
        cur_code = code_map.get(char, '')
        if cur_code and cur_code != prev_code:
            coded += cur_code
        # Per spec, H and W don't reset the previous code group
        if char not in ('H', 'W'):
            prev_code = cur_code

    coded = coded[:4].ljust(4, '0')
    return coded

def detect_entity_type(name: str) -> str:
    """Detect the type of entity based on name patterns."""
    name_upper = name.upper()
    
    for entity_type, patterns in ENTITY_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, name_upper):
                return entity_type
    
    return 'OTHER'

def generate_ngrams(text: str, n: int = 3) -> set:
    """Generate character n-grams for blocking."""
    text = text.lower().replace(' ', '')
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}

def ngram_similarity(text1: str, text2: str, n: int = 3) -> float:
    """Jaccard similarity of n-grams (unweighted baseline)."""
    ngrams1 = generate_ngrams(text1, n)
    ngrams2 = generate_ngrams(text2, n)
    
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    
    return intersection / union if union > 0 else 0.0


def tfidf_ngram_similarity(
    text1: str,
    text2: str,
    idf_weights: Optional[Dict[str, float]] = None,
    n: int = 3,
) -> float:
    """
    TF-IDF weighted cosine similarity of character n-gram vectors.

    If *idf_weights* is ``None`` falls back to plain Jaccard similarity so
    it can be used as a drop-in replacement before an IDF table is built.

    Parameters
    ----------
    text1, text2 : str
        The two strings to compare.
    idf_weights : dict, optional
        Mapping ``{ngram: idf_value}``.  Obtain from
        ``TFIDFNGramIndex._idf`` after calling ``freeze()``.
    n : int
        N-gram size (default 3).
    """
    if idf_weights is None:
        return ngram_similarity(text1, text2, n)

    ng1 = generate_ngrams(text1, n)
    ng2 = generate_ngrams(text2, n)
    if not ng1 or not ng2:
        return 0.0

    def _vec(ngrams: set) -> Dict[str, float]:
        tf = 1.0 / len(ngrams)
        return {ng: tf * idf_weights.get(ng, 1.0) for ng in ngrams}

    v1 = _vec(ng1)
    v2 = _vec(ng2)
    shared = set(v1) & set(v2)
    if not shared:
        return 0.0

    dot = sum(v1[k] * v2[k] for k in shared)
    n1 = math.sqrt(sum(v * v for v in v1.values()))
    n2 = math.sqrt(sum(v * v for v in v2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)

def extract_acronym(name: str) -> str:
    """Extract potential acronym from name."""
    words = name.upper().split()
    skip_words = {'THE', 'OF', 'AND', 'A', 'AN', 'IN', 'FOR', 'TO', 'BY'}
    meaningful_words = [w for w in words if w not in skip_words]
    return ''.join(w[0] for w in meaningful_words if w)

def is_acronym_match(short_name: str, long_name: str) -> bool:
    """Check if short_name could be an acronym of long_name."""
    short = short_name.upper().replace(' ', '').replace('.', '')
    acronym = extract_acronym(long_name)
    
    if len(short) <= 5 and len(short) >= 2:
        return short == acronym or short in acronym or acronym.startswith(short)
    return False

def compute_match_score(
        name1: str,
        name2: str,
        embedding_sim: float,
        phonetic_weight: float = 0.1,
        ngram_weight: float = 0.15,
        fuzzy_weight: float = 0.25,
        embed_weight: float = 0.4,
        acronym_weight: float = 0.1,
        idf_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, dict]:
    """Compute weighted match score using multiple signals.

    Parameters
    ----------
    idf_weights : dict, optional
        If provided, n-gram similarity uses TF-IDF cosine instead of
        plain Jaccard.  Pass ``TFIDFNGramIndex._idf`` here.
    """
    
    soundex1, soundex2 = soundex(name1), soundex(name2)
    phonetic_score = 1.0 if soundex1 == soundex2 else 0.0
    
    ngram_score = tfidf_ngram_similarity(name1, name2, idf_weights=idf_weights, n=3)
    
    fuzzy_scores = [
        fuzz.ratio(name1, name2) / 100.0,
        fuzz.partial_ratio(name1, name2) / 100.0,
        fuzz.token_sort_ratio(name1, name2) / 100.0,
        fuzz.token_set_ratio(name1, name2) / 100.0,
    ]
    fuzzy_score = max(fuzzy_scores)
    
    acronym_score = 1.0 if (is_acronym_match(name1, name2) or is_acronym_match(name2, name1)) else 0.0
    
    total_score = (
            phonetic_weight * phonetic_score +
            ngram_weight * ngram_score +
            fuzzy_weight * fuzzy_score +
            embed_weight * embedding_sim +
            acronym_weight * acronym_score
    )
    
    breakdown = {
        'phonetic': phonetic_score,
        'ngram': ngram_score,
        'fuzzy': fuzzy_score,
        'embedding': embedding_sim,
        'acronym': acronym_score,
        'total': total_score
    }
    
    return total_score, breakdown

def get_meaningful_words(name: str) -> set:
    """
    Extract meaningful distinguishing words from a name.
    
    Two-tier exclusion system:
    1. Entity type suffixes (BANK, CORP, INC) - ALWAYS excluded
    2. Generic prefixes (FIRST, AMERICAN) - excluded only if meaningful words exist
    
    Examples:
    - "STEARNS BANK" -> {'stearns'}
    - "AMERICAN HERITAGE BANK" -> {'heritage'} (american/bank excluded, heritage kept)
    - "US BANK" -> {'us'} (bank always excluded, us kept as fallback)
    - "FIRST BANK" -> {'first'} (bank always excluded, first kept as fallback)
    - "FIRST CHOICE BANK" -> {'choice'} (bank excluded, first excluded, choice kept)
    - "FIRST AMERICAN BANK" -> {'first', 'american'} (bank excluded, both kept as fallback)
    """
    # Tier 1: Entity type words - ALWAYS excluded (never meaningful)
    entity_type_words = {
        'bank', 'banking', 'trust', 'savings', 'credit', 'union',
        'company', 'corporation', 'corp', 'inc', 'llc', 'ltd', 'limited',
        'the', 'of', 'and', 'a', 'an',
        'financial', 'services', 'group', 'holdings', 'capital', 'partners',
        'investment', 'investments', 'management', 'fund', 'funds',
        'advisory', 'advisors', 'asset', 'assets', 'lending', 'loan',
        'equipment', 'finance', 'mortgage', 'solutions', 'associates',
        'na', 'n.a.', 'fsb', 'ssb',
    }
    
    # Tier 2: Generic prefixes - excluded only if there are other meaningful words
    generic_prefixes = {
        'american', 'united', 'us', 'usa', 'new', 'old', 'central',
        'western', 'eastern', 'northern', 'southern', 'north', 'south',
        'east', 'west', 'pacific', 'atlantic', 'continental', 'general',
        'royal', 'premier', 'global', 'international', 'universal',
        'great', 'greater', 'community', 'regional', 'state', 'metro',
        'metropolitan', 'citizens', 'peoples', 'security', 'commerce',
        'first', 'second', 'third', 'national', 'federal',
    }
    
    word_list = name.lower().split()
    
    # Remove entity type words (always)
    words_without_entity_types = [w for w in word_list if w not in entity_type_words and len(w) > 1]
    
    # Check if there are meaningful words after removing BOTH tiers
    all_generic = entity_type_words | generic_prefixes
    truly_meaningful = [w for w in word_list if w not in all_generic and len(w) > 1]
    
    if truly_meaningful:
        # We have truly unique words - return only those
        return set(truly_meaningful)
    
    # No truly unique words - return the non-entity-type words (which are the generic prefixes)
    # These become meaningful by default (e.g., "FIRST AMERICAN BANK" -> {'first', 'american'})
    if words_without_entity_types:
        return set(words_without_entity_types)
    
    return set()

def combined_fuzzy_score(name1: str, name2: str) -> float:
    """
    Compute weighted ensemble of fuzzy scores with safeguards against
    common-word-only matches (e.g. "Heritage Bank" matching "Old National Bank").
    """
    
    ratio = fuzz.ratio(name1, name2)
    partial = fuzz.partial_ratio(name1, name2)
    token_sort = fuzz.token_sort_ratio(name1, name2)
    token_set = fuzz.token_set_ratio(name1, name2)

    # Weighted combination heavily favoring full string similarity (token_sort & ratio)
    # token_set is dangerous because it gives 100% for subset matches
    score = 0.3 * ratio + 0.1 * partial + 0.5 * token_sort + 0.1 * token_set

    # GUARDRAIL 1: Meaningful Word Overlap
    # If strings have meaningful words but share NONE, this is almost certainly wrong
    words1 = get_meaningful_words(name1)
    words2 = get_meaningful_words(name2)
    
    if words1 and words2 and not (words1 & words2):
        # No shared meaningful words (e.g. "AMERICAN MONTANA" vs "AMERICAN")
        # This is a FALSE POSITIVE - crush it
        score *= 0.1  # Severe penalty - almost certain false positive
    elif words1 and words2:
        # Calculate overlap ratio for partial penalty
        overlap_ratio = len(words1 & words2) / len(words1 | words2)
        if overlap_ratio < 0.3:  # Less than 30% meaningful word overlap
            score *= 0.5  # Moderate penalty

    # ENHANCED GUARDRAIL 2: Entity Name Mismatch Detection
    # Special handling for bank names, company names, etc.
    name1_clean = name1.upper().strip()
    name2_clean = name2.upper().strip()

    # Extract primary entity identifiers
    # Strategy: Skip common prefixes ONLY if there's a meaningful word after
    def get_primary_identifier(name):
        words = name.split()
        if not words:
            return None
            
        # Common prefixes we'd like to skip IF there's something better after
        skippable_prefixes = {
            'THE', 'A', 'AN',
            'AMERICAN', 'UNITED', 'US', 'USA', 'NEW', 'OLD', 'CENTRAL',
            'WESTERN', 'EASTERN', 'NORTHERN', 'SOUTHERN', 'NORTH', 'SOUTH',
            'EAST', 'WEST', 'PACIFIC', 'ATLANTIC', 'CONTINENTAL', 'GENERAL',
            'ROYAL', 'PREMIER', 'GLOBAL', 'INTERNATIONAL', 'UNIVERSAL',
            'GREAT', 'GREATER', 'COMMUNITY', 'REGIONAL', 'STATE', 'METRO',
            'METROPOLITAN', 'CITIZENS', 'PEOPLES', 'SECURITY', 'COMMERCE',
            'FIRST', 'SECOND', 'THIRD',
        }
        generic_words = {
            'BANK', 'BANKING', 'CORP', 'CORPORATION', 'INC', 'LLC', 'LTD', 'LIMITED',
            'COMPANY', 'CO', 'TRUST', 'FINANCIAL', 'SERVICES', 'GROUP',
            'EQUIPMENT', 'FINANCE', 'CAPITAL', 'INVESTMENT', 'FUND',
            'NATIONAL', 'FEDERAL', 'SAVINGS', 'CREDIT', 'UNION',
            'LENDING', 'LOAN', 'MORTGAGE', 'HOLDINGS', 'PARTNERS',
        }
        all_generic = skippable_prefixes | generic_words

        # Find first meaningful word
        for word in words:
            if word not in all_generic and len(word) > 2:
                return word
        
        # No meaningful word - return first non-article word
        for word in words:
            if word not in {'THE', 'A', 'AN'} and len(word) > 2:
                return word
        return None

    primary1 = get_primary_identifier(name1_clean)
    primary2 = get_primary_identifier(name2_clean)

    # If both have clear primary identifiers that are completely different
    if primary1 and primary2 and primary1 != primary2:
        # Check if they're similar enough (e.g., abbreviations)
        identifier_sim = fuzz.ratio(primary1, primary2)
        if identifier_sim < 60:  # Primary identifiers are very different
            # Special cases: acronyms, abbreviations
            if not (is_acronym_match(primary1, primary2) or is_acronym_match(primary2, primary1)):
                score *= 0.2  # Heavy penalty for different primary identifiers

        # GUARDRAIL 3: Length mismatch penalty (existing)
    len1, len2 = len(name1), len(name2)
    if len1 > 0 and len2 > 0:
        length_ratio = min(len1, len2) / max(len1, len2)
        if length_ratio < 0.5:
            score *= 0.7

    # GUARDRAIL 4: Very short strings (existing)
    if min(len1, len2) <= 4 and ratio < 80:
        score *= 0.5

    # GUARDRAIL 5: Token set ratio is too high compared to other scores
    # This catches cases where token_set gives false confidence
    if token_set > 90 and ratio < 70:
        # Token set is giving false confidence due to shared common words
        score *= 0.6

    return score


def has_shared_meaningful_identifier(name1: str, name2: str) -> bool:
    """
    Check if two entity names share sufficient meaningful identifying words.
    
    Logic:
    1. If words are identical, match
    2. If one set is a subset AND the subset contains a unique identifier (not just generic), match
    3. Otherwise require > 50% overlap in both directions
    
    A "unique identifier" is a word that's not in the generic_prefix list (truly distinguishing).
    """
    words1 = get_meaningful_words(name1)
    words2 = get_meaningful_words(name2)

    # If EITHER name has no meaningful words, they should NOT match
    if not words1 or not words2:
        return False

    shared = words1 & words2
    
    # No shared words = definitely no match
    if not shared:
        return False
    
    # If identical, definitely match
    if words1 == words2:
        return True
    
    # Generic prefixes that aren't unique enough to anchor a subset match
    generic_prefixes = {
        'american', 'united', 'us', 'usa', 'new', 'old', 'central',
        'western', 'eastern', 'northern', 'southern', 'north', 'south',
        'east', 'west', 'pacific', 'atlantic', 'continental', 'general',
        'royal', 'premier', 'global', 'international', 'universal',
        'great', 'greater', 'community', 'regional', 'state', 'metro',
        'metropolitan', 'citizens', 'peoples', 'security', 'commerce',
        'first', 'second', 'third', 'national', 'federal',
    }
    
    # Check if one is a subset of the other
    smaller, larger = (words1, words2) if len(words1) <= len(words2) else (words2, words1)
    
    if smaller <= larger:
        # Subset detected - only allow if the smaller set contains a unique word
        unique_words_in_smaller = smaller - generic_prefixes
        if unique_words_in_smaller:
            # Has at least one truly unique identifier - allow subset match
            return True
        # All words in smaller set are generic - don't allow subset match
        # (e.g., {'first'} ⊂ {'first', 'american'} but 'first' is generic)
    
    # Otherwise require > 50% overlap in both directions
    overlap1 = len(shared) / len(words1)
    overlap2 = len(shared) / len(words2)
    
    return overlap1 > 0.5 and overlap2 > 0.5


def extract_entity_core(name: str) -> str:
    """
    Extract the core identifying part of an entity name.
    
    Strategy: Skip common prefixes ONLY if there's a meaningful word after them.
    
    Examples:
    - "STEARNS BANK EQUIPMENT FINANCE" -> "STEARNS"
    - "AMERICAN HERITAGE NATIONAL BANK" -> "HERITAGE" (skip AMERICAN, HERITAGE is meaningful)
    - "AMERICAN EXPRESS" -> "EXPRESS" (skip AMERICAN, EXPRESS is meaningful)
    - "US BANK" -> "US" (keep US because BANK is generic - nothing meaningful after)
    - "FIRST BANK" -> "FIRST" (keep FIRST because BANK is generic)
    - "FIRST CHOICE BANK" -> "CHOICE" (skip FIRST, CHOICE is meaningful)
    - "FIRST CITIZENS BANK" -> "FIRST" (CITIZENS is also generic, so keep FIRST)
    """
    words = name.upper().split()
    if not words:
        return name

    # Common prefixes we'd like to skip IF there's something better after
    skippable_prefixes = {
        'THE', 'A', 'AN',
        # Common geographic/generic prefixes
        'AMERICAN', 'UNITED', 'US', 'USA', 'NEW', 'OLD', 'CENTRAL',
        'WESTERN', 'EASTERN', 'NORTHERN', 'SOUTHERN', 'NORTH', 'SOUTH',
        'EAST', 'WEST', 'PACIFIC', 'ATLANTIC', 'CONTINENTAL', 'GENERAL',
        'ROYAL', 'PREMIER', 'GLOBAL', 'INTERNATIONAL', 'UNIVERSAL',
        'GREAT', 'GREATER', 'COMMUNITY', 'REGIONAL', 'STATE', 'METRO',
        'METROPOLITAN', 'CITIZENS', 'PEOPLES', 'SECURITY', 'COMMERCE',
        'FIRST', 'SECOND', 'THIRD',
    }

    # Common suffixes that are too generic to be core identifiers
    generic_words = {
        'BANK', 'BANKING', 'CORP', 'CORPORATION', 'INC', 'LLC', 'LTD', 'LIMITED',
        'COMPANY', 'CO', 'TRUST', 'FINANCIAL', 'SERVICES', 'GROUP',
        'EQUIPMENT', 'FINANCE', 'CAPITAL', 'INVESTMENT', 'FUND',
        'NATIONAL', 'FEDERAL', 'STATE', 'COMMUNITY', 'COMMERCIAL',
        'SAVINGS', 'CREDIT', 'UNION', 'LENDING', 'LOAN', 'MORTGAGE',
        'HOLDINGS', 'PARTNERS', 'ASSOCIATES', 'SOLUTIONS',
    }

    # Combine both sets for "too generic" check
    all_generic = skippable_prefixes | generic_words

    def is_meaningful(word: str) -> bool:
        """Check if a word is meaningful (not generic) and long enough."""
        return word not in all_generic and len(word) > 1

    # Find the first meaningful word in the full name
    first_meaningful = None
    for word in words:
        if is_meaningful(word):
            first_meaningful = word
            break

    # If we found a meaningful word, return it
    if first_meaningful:
        return first_meaningful

    # No meaningful word found - return first TWO non-article words as compound
    # (e.g., "FIRST AMERICAN BANK" -> "FIRST AMERICAN", "US BANK" -> "US BANK")
    fallback_words = []
    for word in words:
        if word not in {'THE', 'A', 'AN'} and len(word) > 1:
            fallback_words.append(word)
            if len(fallback_words) >= 2:
                break
    
    if fallback_words:
        return ' '.join(fallback_words)
    
    return words[0] if words else name


def min_token_overlap(name1: str, name2: str, min_overlap: int = 1) -> bool:
    """
    Check if two names share at least min_overlap tokens.
    Helps filter out false positives from embedding similarity.
    """
    # Get meaningful tokens (exclude very short words)
    tokens1 = {t for t in name1.lower().split() if len(t) > 2}
    tokens2 = {t for t in name2.lower().split() if len(t) > 2}
    return len(tokens1 & tokens2) >= min_overlap

def entity_types_compatible(type1: str, type2: str) -> bool:
    """
    Check if two entity types are compatible for matching.
    Returns False for clear mismatches (e.g., BANK vs MANUFACTURER).
    """
    # If either is OTHER, allow the match
    if type1 == 'OTHER' or type2 == 'OTHER':
        return True

    # Same type is always compatible
    if type1 == type2:
        return True

    # Define compatible type pairs
    compatible_pairs = {
        ('BANK', 'LEGAL'),      # Banks often act as trustees
        ('INVESTMENT', 'LEGAL'), # Investment firms as custodians
    }

    pair = (type1, type2) if type1 < type2 else (type2, type1)
    return pair in compatible_pairs

def should_merge_clusters(label_a: str, label_b: str, embedding_sim: float) -> bool:
    """
    Determine if two cluster labels should be merged.
    VERY CONSERVATIVE: Requires BOTH high embedding similarity AND high fuzzy similarity.
    This prevents merging semantically different entities.
    """
    
    a_lower = label_a.lower().strip()
    b_lower = label_b.lower().strip()

    # Always require fuzzy similarity check to prevent bad merges
    fuzzy_ratio = fuzz.ratio(a_lower, b_lower)
    fuzzy_token_set = fuzz.token_set_ratio(a_lower, b_lower)
    fuzzy_partial = fuzz.partial_ratio(a_lower, b_lower)

    # VERY HIGH embedding similarity (>=0.92) - still require strong fuzzy match
    if embedding_sim >= 0.92:
        # Require at least 80% fuzzy similarity
        if fuzzy_token_set >= 80 and fuzzy_ratio >= 60:
            return True
        return False

    # HIGH embedding similarity (>=0.88) - require very strong fuzzy match
    if embedding_sim >= 0.88:
        # Require both high token set AND partial ratio
        if fuzzy_token_set >= 85 and fuzzy_partial >= 85:
            return True
        # Substring containment only for longer, meaningful strings
        if len(a_lower) >= 10 and len(b_lower) >= 10:
            if a_lower in b_lower or b_lower in a_lower:
                if fuzzy_ratio >= 70:  # Additional guard
                    return True
        return False

    # Below 0.88 embedding similarity - VERY strict requirements
    if embedding_sim >= 0.85:
        # Only merge if fuzzy scores are very high
        if fuzzy_token_set >= 90 and fuzzy_partial >= 90 and fuzzy_ratio >= 75:
            return True
        return False

    # Below 0.85 - don't merge based on embeddings alone
    # Only consider exact substring matches for well-formed names
    if embedding_sim >= 0.80:
        min_len = min(len(a_lower), len(b_lower))
        # Only if shorter is substantial and contained in longer
        if min_len >= 15 and (a_lower in b_lower or b_lower in a_lower):
            # And they share most words
            words_a = set(a_lower.split())
            words_b = set(b_lower.split())
            overlap = len(words_a & words_b)
            total = len(words_a | words_b)
            if total > 0 and overlap / total >= 0.7:
                return True

    return False

