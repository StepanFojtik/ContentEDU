# scripts/finetune.py

from datasets import load_dataset
from loguru import logger
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from trl import SFTTrainer, SFTConfig


def prepare_dataset(tokenizer, dataset_name="mlabonne/FineTome-100k", split="train", chat_template="gemma-3"):
    """
    Loads and formats the dataset using chat templates for finetuning.
    """
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    dataset = standardize_data_formats(dataset)

    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    def apply_chat_template(example):
        return {"text": tokenizer.apply_chat_template(example["conversations"])}

    dataset = dataset.map(apply_chat_template, batched=True)
    return dataset, tokenizer


def train_model(model, tokenizer, dataset, max_steps=30):
    """
    Fine-tunes the model on the dataset using LoRA adapters.
    """
    logger.info("Preparing trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=max_steps,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none"
        )
    )

    logger.info("Applying masking to only train on assistant responses.")
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n"
    )

    logger.info("Starting training...")
    trainer_stats = trainer.train()
    return trainer, trainer_stats


def save_model(model, tokenizer, output_dir="gemma-3-finetune"):
    """
    Saves the finetuned model and tokenizer locally.
    """
    logger.info(f"Saving model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
