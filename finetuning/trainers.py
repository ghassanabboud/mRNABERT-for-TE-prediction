import torch
from transformers import Trainer


class MaskedRegressionTrainer(Trainer):
    """MSE trainer that masks NaN label positions and optionally injects bio_prior_bias.

    Works for both the standard fine-tuning model (bio_prior absent from batch) and
    the bio-prior attention model (bio_prior present): bio_prior_bias is forwarded to
    the model only when it is actually in the batch, so the standard HF model never
    receives an unexpected kwarg.
    """

    _bio_prior_logged: bool = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        bio_prior = inputs.pop("bio_prior", None)

        if not self._bio_prior_logged:
            if bio_prior is not None:
                print("[MaskedRegressionTrainer] bio_prior_bias is being forwarded to the model.")
            else:
                print("[MaskedRegressionTrainer] No bio_prior in batch — running without structural prior.")
            self._bio_prior_logged = True

        if bio_prior is not None:
            inputs["bio_prior_bias"] = bio_prior

        outputs = model(**inputs)
        logits = outputs.logits

        mask = ~torch.isnan(labels)
        loss = torch.nn.functional.mse_loss(logits[mask], labels[mask], reduction="mean")

        return (loss, outputs) if return_outputs else loss
