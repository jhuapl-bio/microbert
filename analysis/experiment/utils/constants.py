# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC

# Available models allowed and tested in pipeline

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