from dnaset.bigwig_dataset import tile_genome
from pyfaidx import Fasta
import sys
import locs
import random
fa_file = Fasta(locs.HUMAN_FA)

tile_genome(
    sequence_length=512,
    reference_fasta=fa_file,
    gap_bed_list=locs.HG38_GAPS,
    shuffle=True,
    chrom_ignore_chars="_M",
    out_path=locs.TILED_BED
)

testfp = open(locs.TEST_BED,'w')
trainfp = open(locs.TRAIN_BED,'w')
sourcefp = open(locs.TILED_BED,'r')
line = sourcefp.readline()
while line != '':
    if random.random() > 0.1:
        trainfp.write(line)
    else:
        testfp.write(line)
    line = sourcefp.readline()
sourcefp.close()

testfp.close()
trainfp.close()


