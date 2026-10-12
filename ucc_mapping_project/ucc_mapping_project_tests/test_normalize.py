# =============================================================================
# Unit tests for normalize_name and related utility functions.
# Run with:  pytest ucc_mapping_project_tests/test_normalize.py -v
# =============================================================================
import pytest
from ucc_mapping_project.utils import (
    pre_clean_name,
    normalize_name,
    soundex,
    detect_entity_type,
    generate_ngrams,
    ngram_similarity,
    extract_entity_core,
    get_meaningful_words,
    has_shared_meaningful_identifier,
    is_acronym_match,
    entity_types_compatible,
    combined_fuzzy_score,
    min_token_overlap,
)


# =========================================================================
#  normalize_name
# =========================================================================

class TestNormalizeName:
    """Core normalization tests — the most fragile function in the pipeline."""

    # --- basic cleaning ---
    def test_strips_whitespace(self):
        assert normalize_name("  WELLS FARGO  ") == "WELLS FARGO"

    def test_uppercases(self):
        assert normalize_name("wells fargo") == "WELLS FARGO"

    def test_empty_and_nan(self):
        assert normalize_name("") == ""
        assert normalize_name(None) == ""
        assert normalize_name("   ") == ""
        assert normalize_name(float("nan")) == ""

    # --- legal suffix stripping ---
    def test_strips_inc(self):
        result = normalize_name("ACME CORP INC")
        assert "INC" not in result

    def test_strips_llc(self):
        result = normalize_name("ACME LLC")
        assert "LLC" not in result

    def test_strips_corporation(self):
        result = normalize_name("GENERAL MOTORS CORPORATION")
        assert "CORPORATION" not in result

    def test_strips_limited(self):
        result = normalize_name("BARCLAYS LIMITED")
        assert "LIMITED" not in result

    def test_strips_na_suffix(self):
        """N.A. is a legal suffix (National Association)."""
        result = normalize_name("WELLS FARGO BANK N.A.")
        assert "N A" not in result  # after punctuation strip it would be "N A"
        assert "WELLS FARGO BANK" in result

    # --- BANK is intentionally KEPT ---
    def test_keeps_bank(self):
        result = normalize_name("STEARNS BANK")
        assert "BANK" in result

    def test_keeps_banking(self):
        result = normalize_name("ONLINE BANKING SERVICES")
        # BANKING should survive normalization
        assert "BANK" in result or "BANKING" in result

    def test_keeps_credit_union(self):
        result = normalize_name("NAVY FEDERAL CREDIT UNION")
        assert "CREDIT" in result

    # --- role qualifier stripping ---
    def test_strips_as_trustee(self):
        result = normalize_name("JPMORGAN CHASE BANK, AS TRUSTEE FOR XYZ")
        assert "TRUSTEE" not in result
        assert "JPMORGAN CHASE BANK" in result

    def test_strips_as_agent(self):
        result = normalize_name("CITIBANK, AS ADMINISTRATIVE AGENT")
        assert "AGENT" not in result
        assert "CITIBANK" in result

    def test_strips_on_behalf_of(self):
        result = normalize_name("BANK OF AMERICA, ON BEHALF OF INVESTORS")
        assert "BEHALF" not in result

    # --- common word removal ---
    def test_strips_the(self):
        result = normalize_name("THE BANK OF NEW YORK MELLON")
        # "THE" should be gone; "OF" might or might not be removed
        assert not result.startswith("THE ")

    def test_strips_dba(self):
        result = normalize_name("ACME CORP DBA ACME LENDING")
        assert "DBA" not in result

    def test_strips_fka(self):
        result = normalize_name("NEWCO FKA OLDCO")
        assert "FKA" not in result

    def test_strips_formerly_known_as(self):
        result = normalize_name("NEWCO FORMERLY KNOWN AS OLDCO")
        assert "FORMERLY" not in result
        assert "KNOWN" not in result

    # --- A and AN should NOT be stripped (regression) ---
    def test_does_not_strip_a_inside_name(self):
        """Ensure 'A' and 'AN' are not accidentally stripped from names.
        Previously \bA\b matched the isolated 'A' in 'N A' after
        punctuation stripping of 'N.A.'."""
        # This name should keep its structure; 'A' inside a word is fine
        result = normalize_name("AMERICAN SAVINGS BANK")
        assert "AMERICAN" in result

    # --- abbreviation expansion ---
    def test_expands_ntl(self):
        result = normalize_name("NTL CITY BANK")
        assert "NATIONAL" in result
        assert "NTL" not in result

    def test_expands_bk(self):
        result = normalize_name("FIRST BK")
        assert "BANK" in result
        assert "BK" not in result

    def test_expands_intl(self):
        result = normalize_name("INTL FINANCE")
        assert "INTERNATIONAL" in result

    # --- punctuation handling ---
    def test_strips_punctuation(self):
        result = normalize_name("J.P. MORGAN CHASE & CO.")
        # All dots and ampersands should become spaces
        assert "." not in result
        assert "&" not in result

    def test_collapses_whitespace(self):
        result = normalize_name("WELLS   FARGO    BANK")
        assert "  " not in result

    # --- idempotency ---
    def test_normalize_idempotent(self):
        """Normalizing an already-normalized name should return the same result."""
        first = normalize_name("JPMORGAN CHASE BANK, AS TRUSTEE")
        second = normalize_name(first)
        assert first == second


# =========================================================================
#  soundex
# =========================================================================

class TestSoundex:
    def test_basic(self):
        assert soundex("Robert") == "R163"

    def test_hw_separator_rule(self):
        """ASHCRAFT and ASHCROFT should produce the same Soundex code."""
        assert soundex("ASHCRAFT") == soundex("ASHCROFT")

    def test_empty(self):
        assert soundex("") == ""

    def test_all_vowels(self):
        assert soundex("AEIOU") == "A000"

    def test_short_name(self):
        assert soundex("AL") == "A400"


# =========================================================================
#  detect_entity_type
# =========================================================================

class TestDetectEntityType:
    def test_bank(self):
        assert detect_entity_type("WELLS FARGO BANK") == "BANK"

    def test_investment(self):
        assert detect_entity_type("BLACKROCK CAPITAL PARTNERS") == "INVESTMENT"

    def test_manufacturer(self):
        assert detect_entity_type("CATERPILLAR EQUIPMENT") == "MANUFACTURER"

    def test_government(self):
        assert detect_entity_type("U.S. DEPARTMENT OF AGRICULTURE") == "GOVERNMENT"

    def test_other(self):
        assert detect_entity_type("ACME WIDGETS") == "OTHER"


# =========================================================================
#  n-gram helpers
# =========================================================================

class TestNGrams:
    def test_generate_basic(self):
        ngrams = generate_ngrams("hello", 3)
        assert "hel" in ngrams
        assert "ell" in ngrams
        assert "llo" in ngrams

    def test_short_text(self):
        ngrams = generate_ngrams("ab", 3)
        assert ngrams == {"ab"}

    def test_similarity_identical(self):
        assert ngram_similarity("WELLS FARGO", "WELLS FARGO") == 1.0

    def test_similarity_different(self):
        assert ngram_similarity("WELLS FARGO", "BANK OF AMERICA") < 0.3


# =========================================================================
#  entity_types_compatible
# =========================================================================

class TestEntityTypesCompatible:
    def test_same_type(self):
        assert entity_types_compatible("BANK", "BANK") is True

    def test_other_allows(self):
        assert entity_types_compatible("OTHER", "BANK") is True

    def test_incompatible(self):
        assert entity_types_compatible("BANK", "MANUFACTURER") is False

    def test_bank_legal_compatible(self):
        assert entity_types_compatible("BANK", "LEGAL") is True


# =========================================================================
#  extract_entity_core / get_meaningful_words / has_shared_meaningful_identifier
# =========================================================================

class TestEntityHelpers:
    def test_extract_core(self):
        assert extract_entity_core("STEARNS BANK EQUIPMENT FINANCE") == "STEARNS"

    def test_extract_core_strips_prefix(self):
        # After fix: THE is skipped. No meaningful word exists (FIRST, NATIONAL, BANK all generic)
        # So we return first TWO non-article words as compound: "FIRST NATIONAL"
        assert extract_entity_core("THE FIRST NATIONAL BANK") == "FIRST NATIONAL"

    def test_meaningful_words_excludes_bank(self):
        words = get_meaningful_words("STEARNS BANK")
        assert "stearns" in words
        assert "bank" not in words

    def test_shared_identifier_true(self):
        assert has_shared_meaningful_identifier("STEARNS BANK", "STEARNS FINANCIAL") is True

    def test_shared_identifier_false(self):
        # After fix: "US BANK" has no meaningful words (US and BANK both excluded)
        # So this returns False (no meaningful overlap possible)
        assert has_shared_meaningful_identifier("STEARNS BANK", "US BANK") is False


# =========================================================================
#  is_acronym_match
# =========================================================================

class TestAcronymMatch:
    def test_ibm(self):
        assert is_acronym_match("IBM", "INTERNATIONAL BUSINESS MACHINES") is True

    def test_not_acronym(self):
        assert is_acronym_match("LONGNAME", "INTERNATIONAL BUSINESS MACHINES") is False

    def test_too_long(self):
        assert is_acronym_match("TOOLONG", "TEST OPERATING LONG") is False


# =========================================================================
#  min_token_overlap
# =========================================================================

class TestMinTokenOverlap:
    def test_overlap(self):
        assert min_token_overlap("WELLS FARGO BANK", "WELLS FARGO FINANCIAL") is True

    def test_no_overlap(self):
        assert min_token_overlap("ALPHA CORP", "BETA SYSTEMS") is False


# =========================================================================
#  combined_fuzzy_score
# =========================================================================

class TestCombinedFuzzyScore:
    def test_identical_names_high(self):
        score = combined_fuzzy_score("WELLS FARGO BANK", "WELLS FARGO BANK")
        assert score >= 95.0

    def test_different_names_low(self):
        # Completely different entities should score low
        score = combined_fuzzy_score("ACME WIDGETS INC", "BETA SYSTEMS LLC")
        assert score < 40.0, f"False positive: score={score}"

    def test_similar_names_moderate(self):
        score = combined_fuzzy_score("WELLS FARGO", "WELLS FARGO BANK")
        assert 50.0 < score < 100.0


# =========================================================================
#  False Positive Prevention Tests
# =========================================================================

class TestFalsePositivePrevention:
    """Tests to ensure common-prefix entities don't falsely match."""

    def test_american_heritage_vs_american_express(self):
        """Regression test: these should NOT match."""
        score = combined_fuzzy_score(
            "AMERICAN HERITAGE NATIONAL BANK",
            "AMERICAN EXPRESS"
        )
        # Score should be heavily penalized due to different core identifiers
        assert score < 20.0, f"False positive: score={score}"

    def test_american_heritage_no_shared_identifier(self):
        """'american' should be excluded as a meaningful word."""
        assert has_shared_meaningful_identifier(
            "AMERICAN HERITAGE NATIONAL BANK",
            "AMERICAN EXPRESS"
        ) is False

    def test_american_heritage_extract_core(self):
        """Core should be HERITAGE, not AMERICAN."""
        assert extract_entity_core("AMERICAN HERITAGE NATIONAL BANK") == "HERITAGE"
        assert extract_entity_core("AMERICAN EXPRESS") == "EXPRESS"

    def test_first_national_vs_first_citizens(self):
        """Two different 'First' banks should NOT match with strict logic."""
        # FIRST NATIONAL BANK -> {'first', 'national'} (both generic, kept as fallback)
        # FIRST CITIZENS BANK -> {'first', 'citizens'} (both generic, kept as fallback)
        # They share 'first' but that's a generic word, so subset match not allowed
        assert has_shared_meaningful_identifier(
            "FIRST NATIONAL BANK",
            "FIRST CITIZENS BANK"
        ) is False  # Different banks

    def test_us_bank_vs_first_bank(self):
        """US BANK and FIRST BANK should NOT match (different distinguishing words)."""
        assert has_shared_meaningful_identifier(
            "US BANK",
            "FIRST BANK"
        ) is False  # {'us'} vs {'first'} - no overlap

    def test_citizens_bank_vs_first_citizens_bank(self):
        """CITIZENS BANK and FIRST CITIZENS BANK are DIFFERENT banks."""
        # CITIZENS BANK -> {'citizens'} (citizens is generic, but only word left after removing 'bank')
        # FIRST CITIZENS BANK -> {'first', 'citizens'} (both generic, kept as fallback)
        assert get_meaningful_words("CITIZENS BANK") == {'citizens'}
        assert get_meaningful_words("FIRST CITIZENS BANK") == {'first', 'citizens'}
        # citizens is in both, but it's a generic word so subset match not allowed
        assert has_shared_meaningful_identifier(
            "CITIZENS BANK",
            "FIRST CITIZENS BANK"
        ) is False  # Different banks
        # Score should be low due to guardrails
        score = combined_fuzzy_score("CITIZENS BANK", "FIRST CITIZENS BANK")
        assert score < 20.0, f"False positive: score={score}"

    def test_united_bank_vs_united_airlines(self):
        """Completely different entities starting with UNITED."""
        assert has_shared_meaningful_identifier(
            "UNITED BANK",
            "UNITED AIRLINES"
        ) is False

    def test_us_bank_has_no_meaningful_words(self):
        """US BANK keeps 'us' as meaningful since it's the only distinguishing word."""
        words = get_meaningful_words("US BANK")
        assert words == {'us'}  # 'us' is kept because BANK is generic

    def test_stearns_bank_vs_us_bank(self):
        """Different banks should not share meaningful identifier."""
        assert has_shared_meaningful_identifier(
            "STEARNS BANK",
            "US BANK"
        ) is False

    def test_wells_fargo_self_match(self):
        """Same entity should still match itself."""
        assert has_shared_meaningful_identifier(
            "WELLS FARGO BANK",
            "WELLS FARGO FINANCIAL"
        ) is True
        score = combined_fuzzy_score("WELLS FARGO BANK", "WELLS FARGO FINANCIAL")
        assert score > 50.0  # Should still be a decent match

    def test_first_american_bank_edge_cases(self):
        """FIRST AMERICAN BANK should not match partial variants."""
        # FIRST AMERICAN BANK -> {'first', 'american'} (both generic, kept as fallback)
        assert get_meaningful_words("FIRST AMERICAN BANK") == {'first', 'american'}
        
        # Should NOT match FIRST BANK (different bank)
        assert has_shared_meaningful_identifier(
            "FIRST AMERICAN BANK", "FIRST BANK"
        ) is False
        
        # Should NOT match AMERICAN BANK (different bank)
        assert has_shared_meaningful_identifier(
            "FIRST AMERICAN BANK", "AMERICAN BANK"
        ) is False
        
        # Should NOT match FIRST AMERICAN TITLE (different company type)
        assert has_shared_meaningful_identifier(
            "FIRST AMERICAN BANK", "FIRST AMERICAN TITLE"
        ) is False
        
        # SHOULD match FIRST AMERICAN BANK NA (same bank, different suffix)
        assert has_shared_meaningful_identifier(
            "FIRST AMERICAN BANK", "FIRST AMERICAN BANK NA"
        ) is True



# ---------------------------------------------------------------------------
# pre_clean_name: structural junk around the lender's name (added 2026-10)
# ---------------------------------------------------------------------------
class TestPreClean:
    @pytest.mark.parametrize("raw,expected", [
        ("STARION BANK (LENDER AND LETTER OF CREDIT ISSUER)", "STARION BANK"),
        ("WELLS FARGO BANK, NATIONAL ASSOCIATION, AS ADMINISTRIVE AGENT", "WELLS FARGO BANK"),
        ("PLAINSCAPITAL BANK, A TEXAS STATE BANKING ASSOCIATION", "PLAINSCAPITAL BANK"),
        ("TD BANK, NA, AS SUCCESSOR BY MERGER TO MERCANTILE BANK", "TD BANK"),
        ("CENTENNIAL BANK F/K/A LIBERTY BANK OF ARKANSAS", "CENTENNIAL BANK"),
        ("KEY EQUIPMENT FINANCE, A DIVISION OF KEYBANK NA", "KEYBANK"),
        ("FEDERAL DEPOSIT INSURANCE CORPORATION AS RECEIVER OF FIRST REPUBLIC BANK, SAN FRANCISCO, CALIFORNIA", "FIRST REPUBLIC BANK"),
        ("PARK BANK- BROOKFIELD", "PARK BANK"),
        ("BANK ONE, NA, WITH ITS MAIN OFFICE IN COLUMBUS, OH", "BANK ONE"),
        ("FIRST-CITIZENS BANK & TRUST COMPANY", "FIRST-CITIZENS BANK & TRUST COMPANY"),
        ("STATE BANK OF TEXAS", "STATE BANK OF TEXAS"),
    ])
    def test_cases(self, raw, expected):
        assert pre_clean_name(raw) == expected

    def test_never_empties_a_name(self):
        assert pre_clean_name("NA") == "NA"
        assert pre_clean_name("(AS AGENT)") == "(AS AGENT)"

    def test_normalize_uses_pre_clean(self):
        assert normalize_name("KEY EQUIPMENT FINANCE, A DIVISION OF KEYBANK NA") == "KEYBANK"
        assert normalize_name("WESTERN ALLIANCE BANK, AS SUCCESSOR-IN-INTEREST TO BRIDGE BANK, N.A.") == "WESTERN ALLIANCE BANK"

    def test_order_swapped_names_stay_below_review_floor(self):
        assert combined_fuzzy_score("NATIONAL CITY BANK", "CITY NATIONAL BANK") < 85
