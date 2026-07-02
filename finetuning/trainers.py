import torch
from transformers import Trainer


class MaskedRegressionTrainer(Trainer):
    """MSE trainer, inherents all logic fro HFTrainer but ovverides compute_loss for two reasons:
    1. MSE loss is computed only on non-NaN label positions, since some sequences have missing labels for some cell types for RiboNN.
    2. Optionally injects a bio_prior_bias tensor into the model forward pass, if present in the batch. This is used for the bio-prior attention model.
    This trainer thus works for both the standard fine-tuning model (bio_prior absent from batch) and the bio-prior attention model (bio_prior present).
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
