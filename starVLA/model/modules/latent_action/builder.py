from starVLA.model.modules.latent_action.interface import BaseLatentActionEncoder


def build_latent_action_encoder(config) -> BaseLatentActionEncoder:
    la_cfg = config.framework.get("latent_action", {})
    backend = str(la_cfg.get("backend", "univla")).lower()
    if backend == "univla":
        from starVLA.model.modules.latent_action.univla_encoder import UniVLALatentActionEncoder

        return UniVLALatentActionEncoder(config)
    if backend in {"unit", "groot_unit"}:
        from starVLA.model.modules.latent_action.unit_encoder import UniTVisualLatentActionEncoder

        return UniTVisualLatentActionEncoder(config)
    raise NotImplementedError(f"Latent-action backend `{backend}` is not implemented.")
