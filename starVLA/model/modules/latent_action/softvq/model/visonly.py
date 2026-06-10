from torch import nn

from starVLA.model.modules.latent_action.softvq.model.visual_vqvae import VisualVQVAE


class VisOnly(nn.Module):
    def __init__(self, config):
        super().__init__()
        visual_config = config.get("visual_model", config)
        self.visual = VisualVQVAE(visual_config)

    def forward(self, batch):
        return self.visual(batch["source_image"], batch["target_image"])


__all__ = ["VisOnly"]
