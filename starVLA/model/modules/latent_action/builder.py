from starVLA.model.modules.latent_action.interface import BaseLatentActionEncoder


def build_latent_action_encoder(config) -> BaseLatentActionEncoder:
    la_cfg = config.framework.get("latent_action", {})
    backend = str(la_cfg.get("backend", "univla")).lower()
    if backend == "univla":
        from starVLA.model.modules.latent_action.univla_encoder import UniVLALatentActionEncoder

        return UniVLALatentActionEncoder(config)
    if backend == "softvq":
        from starVLA.model.modules.latent_action.softvq_encoder import SoftVQLatentActionEncoder

        return SoftVQLatentActionEncoder(config)
    if backend == "sharla":
        from starVLA.model.modules.latent_action.sharla_encoder import SharlaLatentActionEncoder

        return SharlaLatentActionEncoder(config)
    if backend in {"unit", "groot_unit"}:
        from starVLA.model.modules.latent_action.unit_encoder import UniTVisualLatentActionEncoder

        return UniTVisualLatentActionEncoder(config)
    if backend == "villax":
        from starVLA.model.modules.latent_action.villax_encoder import VillaXLatentActionEncoder

        return VillaXLatentActionEncoder(config)
    raise NotImplementedError(f"Latent-action backend `{backend}` is not implemented.")
