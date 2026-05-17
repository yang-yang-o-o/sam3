"""Local paths for example scripts (avoid HuggingFace gated-repo download)."""
import os

import sam3

SAM3_ROOT = os.path.abspath(os.path.join(os.path.dirname(sam3.__file__), ".."))
BPE_PATH = os.path.join(os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
SAM3_CHECKPOINT = os.path.join(SAM3_ROOT, "checkpoints", "sam3", "sam3.pt")
SAM31_CHECKPOINT = os.path.join(SAM3_ROOT, "checkpoints", "sam3.1", "sam3.1_multiplex.pt")
