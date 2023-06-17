

from dnaset.bigwig_dataset import tile_genome
from pyfaidx import Fasta
import sys
import locs

fa_file = Fasta(locs.HUMAN_FA);
bed_gaps = "hg38.gap"
tile_genome(
sequence_length=512,
reference_fasta=fa_file,
gap_bed_list=bed_gaps,
out_path="hg38_tiled.bed",
shuffle=True,

)