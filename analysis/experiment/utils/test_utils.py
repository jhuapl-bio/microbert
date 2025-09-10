from pathlib import Path
import gzip
from Bio import SeqIO

def parse_input_fasta(path):
    """
    Parses a .fasta or .fastq file (optionally gzipped) and returns a list of SeqRecord objects.
    """
    sequences = []
    p = Path(path)
    suffixes = p.suffixes  # e.g. ['.fastq', '.gz']
    
    is_gz = suffixes and suffixes[-1] == '.gz'
    
    
    fmt_ext = suffixes[-2] if is_gz else suffixes[-1]
    fmt = None
    if fmt_ext in ('.fastq', '.fq', '.FASTQ', '.FQ'):
        fmt = 'fastq'
    elif fmt_ext in ('.fasta', '.fa', '.FA', '.FASTA'):
        fmt = 'fasta'
    else:
        raise ValueError(f"Unrecognized file format: {fmt_ext}")
    
    
    open_fn = gzip.open if is_gz else open
    
    
    with open_fn(path, 'rt') as handle:
        for record in SeqIO.parse(handle, fmt):
            sequences.append(str(record.seq))
    
    return sequences


def normalize_label(label):
    """
    Convert numeric labels to int if they represent whole numbers,
    otherwise return as string.
    """
    if isinstance(label, (int, float)) and float(label).is_integer():
        return int(label)
    return str(label)