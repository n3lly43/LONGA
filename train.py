
import lightning.pytorch as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from models import prepare_model, prepare_dataset, compute_metrics

@hydra_runner(config_path="configs/whisper", config_name="luganda_whisper_salt")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    if cfg.model_arch=='NeMo':
        trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
        exp_manager(trainer, cfg.get("exp_manager", None))
        asr_model = EncDecRNNTBPEModel(cfg=cfg.model, trainer=trainer)

        # Initialize the weights of the model from another model, if provided via config
        asr_model.maybe_init_from_pretrained_checkpoint(cfg)

        trainer.fit(asr_model)

        if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
            if asr_model.prepare_test(trainer):
                trainer.test(asr_model)

    else:
        dataset = prepare_dataset(cfg.dataset)
        model, data_collator, processor = prepare_model(cfg.model)
        training_args = Seq2SeqTrainingArguments(cfg.training_args)

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset[cfg.dataset.eval_set],
            data_collator=data_collator,
            compute_metrics=lambda x: compute_metrics(x, cfg.model.pretrained_model),
            tokenizer=processor.feature_extractor,
        )
        trainer.train()
        trainer.evaluate(dataset['test' if cfg.dataset.eval_set=='val' else 'val'])


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter