# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC


# Constants for models to be used in experiment runs
MODELS = {
    "NT": [
        "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
    ],
    "DNABERT": [
        "zhihan1996/DNABERT-2-117M",
        "zhihan1996/DNABERT-S",
    ],
    "HYENA": [
        "LongSafari/hyenadna-large-1m-seqlen-hf",
        "LongSafari/hyenadna-medium-450k-seqlen-hf",
        "LongSafari/hyenadna-medium-160k-seqlen-hf",
        "LongSafari/hyenadna-small-32k-seqlen-hf",
    ],
    "METAGENE": ["metagene-ai/METAGENE-1"],
    "GenomeOcean": [
        "pGenomeOcean/GenomeOcean-4B",
        "pGenomeOcean/GenomeOcean-100M",
        "pGenomeOcean/GenomeOcean-500M",
    ],
}

# Constants for available taxonomic ranks
TAX_RANKS = [
    "superkingdom",
    "kingdom",
    "phylum",
    "class",
    "family",
    "order",
    "genus",
    "species",
]
