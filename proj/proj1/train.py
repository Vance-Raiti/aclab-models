import trainer
import model
import locs
import dnaset.bigwig_dataset as bigwig_dataset
import torch

import sys

from dnaset.bigwig_dataset import catch_generator_index_error
from dnaset.rbedtool import catch_bed_parse_error



train_dataset = bigwig_dataset.BigWigDataset(
    bigwig_files=locs.BIGWIG_FILE,
    reference_fasta_file=locs.HUMAN_FA,
    input_bed_file=locs.TRAIN_BED
)

valid_dataset = bigwig_dataset.BigWigDataset(
    bigwig_files=locs.BIGWIG_FILE,
    reference_fasta_file=locs.HUMAN_FA,
    input_bed_file=locs.TEST_BED
)

trainconf = trainer.TrainerConfig(
    max_epochs=1,
    batch_size=32,
    ckpt_path="/projectnb/aclab/vraiti/nn/basic_transformer/params/basic_convnet",
    ckpt_interval=100_000
)

mod = model.Model(n_blocks=8,n_steps=2,in_channels=4)
opt_mod = torch.compile(mod)

trainer_object = trainer.Trainer(
    model=opt_mod,
    train_dataset=train_dataset,
    test_dataset=valid_dataset,
    config=trainconf,
    
)

with catch_generator_index_error(), catch_bed_parse_error():
    trainer_object.train(progress_bar=False)
