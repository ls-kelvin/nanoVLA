"""Pre-build VLM processor inputs inside dataloader workers.

The frameworks call ``qwen_vl_interface.build_qwenvl_inputs`` from within
``forward``, so the chat template and the image preprocessing run on the
training process' main thread while the GPU sits idle.  Moving that call into
the collate function keeps it on the dataloader workers instead.
"""

from starVLA.model.modules.vlm.QWen3 import build_qwen3_vl_inputs


class QwenVLInputCollator:
    """Wrap a dataset collate function with Qwen3-VL input construction."""

    def __init__(self, base_collate_fn, base_vlm: str, cot_prompt: str | None = None):
        self.base_collate_fn = base_collate_fn
        self.base_vlm = str(base_vlm)
        self.cot_prompt = cot_prompt
        self._processor = None

    def __getstate__(self):
        # The processor is rebuilt lazily in each worker; never ship it through
        # the worker-start pickle.
        state = self.__dict__.copy()
        state["_processor"] = None
        return state

    def _get_processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(self.base_vlm)
            processor.tokenizer.padding_side = "left"
            self._processor = processor
        return self._processor

    def __call__(self, batch):
        examples = self.base_collate_fn(batch)
        # The payload lives under a key of the first example instead of on the
        # batch object itself: ``accelerate``'s ``send_to_device`` rebuilds the
        # batch with ``type(batch)(...)`` and would drop any attribute, while
        # dict values are recursed into (and moved to the GPU for free).  Only
        # the first example carries it, so ``pin_memory`` pins it once.
        examples[0]["vlm_inputs"] = build_qwen3_vl_inputs(
            self._get_processor(),
            [example["image"] for example in examples],
            [example["lang"] for example in examples],
            cot_prompt=self.cot_prompt,
        )
        return examples


def build_vlm_input_collator(cfg, base_collate_fn):
    """Return a pre-building collator, or ``base_collate_fn`` when disabled."""
    vla_data_cfg = cfg.datasets.vla_data
    if not bool(vla_data_cfg.get("precompute_vlm_inputs", False)):
        return base_collate_fn

    base_vlm = str(cfg.framework.qwenvl.base_vlm)
    if not ("Qwen3-VL" in base_vlm or "rynnbrain" in base_vlm.lower()):
        raise ValueError(
            "datasets.vla_data.precompute_vlm_inputs is only implemented for the "
            f"Qwen3-VL interface, got framework.qwenvl.base_vlm={base_vlm}."
        )
    cot_prompt = vla_data_cfg.get("CoT_prompt", "") if "CoT_prompt" in vla_data_cfg else None
    return QwenVLInputCollator(base_collate_fn, base_vlm, cot_prompt=cot_prompt)
