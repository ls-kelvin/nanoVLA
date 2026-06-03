from abc import ABC, abstractmethod
from typing import Sequence

import torch
import torch.nn as nn
from PIL import Image


class BaseLatentActionEncoder(nn.Module, ABC):
    """Common interface for latent-action pseudo-label encoders."""

    @abstractmethod
    def encode(
        self,
        frame_pairs: Sequence[Sequence[Image.Image]],
        instructions: Sequence[str] | None = None,
    ) -> torch.LongTensor:
        """Encode frame pairs into discrete latent-action indices.

        Args:
            frame_pairs: Flat sequence of ``(source_frame, target_frame)`` PIL pairs.
            instructions: Optional language instructions aligned with pairs.

        Returns:
            Tensor shaped ``[num_pairs, codes_per_pair]``.
        """
        raise NotImplementedError
