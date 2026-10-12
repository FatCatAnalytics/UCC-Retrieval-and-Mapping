# GENERATED FILE - DO NOT EDIT. Source: ucc_mapping_project/ucc_mapping_project/constants.py
# =============================================================================
# CONSTANTS
# =============================================================================
import re

LEGAL_SUFFIXES = [
    r'\bINC\.?$', r'\bINCORPORATED$', r'\bCORPORATION$', r'\bCORP\.?$',
    r'\bLLC\.?$', r'\bL\.?L\.?C\.?$', r'\bLIMITED LIABILITY COMPANY$',
    r'\bL\.?P\.?$', r'\bLIMITED PARTNERSHIP$', r'\bLTD\.?$', r'\bLIMITED$',
    r'\bCO\.?$', r'\bCOMPANY$', r'\bN\.?A\.?$', r'\bNATIONAL ASSOCIATION$',
    r'\bF\.?S\.?B\.?$', r'\bFEDERAL SAVINGS BANK$',
    r'\bS\.?B\.?$', r'\bSAVINGS BANK$', r'\bTRUST COMPANY$',
    r'\bPCA$', r'\bSSB$', r'\bFSB$',
    # NOTE: BANK, BANKING, CREDIT UNION intentionally NOT stripped —
    # they carry entity-type signal needed for matching & clustering.
]

ROLE_TRIGGERS = [
    r"\bsolely\s+in\s+its\s+capacity\s+as\b",
    r"\bin\s+its\s+capacity\s+as\b",
    r"\bas\s+trustee\b",
    r"\bas\s+indenture\s+trustee\b",
    r"\bas\s+administrative\s+agent\b",
    r"\bas\s+collateral\s+agent\b",
    r"\bas\s+security\s+trustee\b",
    r"\bas\s+paying\s+agent\b",
    r"\bas\s+registrar\b",
    r"\bas\s+transfer\s+agent\b",
    r"\bas\s+fiscal\s+agent\b",
    r"\bas\s+custodian\b",
    r"\bas\s+purchaser\s+representative\b",
    r"\bas\s+agent\b",
    r"\bas\s+representative\b",
    r"\bfor\s+itself\s+and\b",
    r"\bon\s+behalf\s+of\b",
]

ROLE_RE = re.compile(r"(?i)\s*(?:,|\s)+(" + "|".join(ROLE_TRIGGERS) + r")\b.*$")

COMMON_WORDS = sorted([
    # Multi-word phrases
    r'\bFORMERLY\s+KNOWN\s+AS\b', r'\bSUCCESSOR\s+BY\s+MERGER\s+OF\b',
    r'\bA\s+DIVISION\s+OF\b', r'\bDIVISION\s+OF\b', r'\bDIVISION\b',
    r'\bD/?B/?A\b', r'\bF/?K/?A\b', r'\bAKA\b',
    # Single words — keep only determiners/prepositions that never carry entity meaning
    r'\bTHE\b', r'\bAND\b', r'\b&\b',
    r'\bAS\b', r'\bBY\b', r'\bFOR\b', r'\bIN\b', r'\bTO\b', r'\bWITH\b',
    # NOTE: \bA\b and \bAN\b removed — too aggressive; they match single-letter
    # tokens that appear after punctuation is stripped (e.g. "N.A." → "N A").
], key=len, reverse=True)  # Longest patterns first → greedy matching

ABBREVIATIONS = {
    r'\bNTL\b': 'NATIONAL',
    r'\bFED\b': 'FEDERAL',
    r'\bCTR\b': 'CENTER',
    r'\bFIN\b': 'FINANCIAL',
    r'\bSVC\b': 'SERVICE',
    r'\bSVCS\b': 'SERVICES',
    r'\bINV\b': 'INVESTMENT',
    r'\bINVS\b': 'INVESTMENTS',
    r'\bMGT\b': 'MANAGEMENT',
    r'\bASSOC\b': 'ASSOCIATION',
    r'\bGRP\b': 'GROUP',
    r'\bCOMM\b': 'COMMERCIAL',
    r'\bBK\b': 'BANK',
    r'\bNATL\b': 'NATIONAL',
    r'\bINTL\b': 'INTERNATIONAL',
    r'\bSVGS\b': 'SAVINGS',
    r'\bEQUIP\b': 'EQUIPMENT',
    r'\bCONST\b': 'CONSTRUCTION',
}

ENTITY_TYPE_PATTERNS = {
    'BANK': [
        r'\bBANK\b', r'\bBANKING\b', r'\bSAVINGS\b', r'\bTRUST\b',
        r'\bN\.?A\.?\b', r'\bF\.?S\.?B\.?\b', r'\bCREDIT UNION\b'
    ],
    'GOVERNMENT': [
        r'\bADMINISTRATION\b', r'\bFEDERAL\b', r'\bU\.?S\.?\b',
        r'\bGOVERNMENT\b', r'\bAGENCY\b', r'\bDEPARTMENT\b'
    ],
    'INVESTMENT': [
        r'\bCAPITAL\b', r'\bPARTNERS\b', r'\bINVESTMENT\b',
        r'\bFUND\b', r'\bADVISOR\b', r'\bASSET\b', r'\bEQUITY\b'
    ],
    'MANUFACTURER': [
        r'\bMANUFACTURING\b', r'\bINDUSTRIES\b', r'\bPRODUCTS\b',
        r'\bEQUIPMENT\b', r'\bMACHINERY\b'
    ],
    'LEGAL': [
        r'\bTRUSTEE\b', r'\bAGENT\b', r'\bCUSTODIAN\b', r'\bLAW\b'
    ]
}