from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.modeling_outputs import SequenceClassifierOutput


@dataclass
class PeptideSequenceClassifierOutput(SequenceClassifierOutput):

    pooler_output: Optional[torch.FloatTensor] = None