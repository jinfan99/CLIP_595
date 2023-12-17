from transformers import CLIPProcessor, CLIPModel, CLIPConfig, VisionTextDualEncoderModel, VisionTextDualEncoderProcessor, AutoTokenizer,AutoImageProcessor
from dataset import CLIPDataset
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import DefaultDataCollator

config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

# model = VisionTextDualEncoderModel.from_vision_text_pretrained(
#     "openai/clip-vit-base-patch32", "roberta-base"
# )

model = CLIPModel(config)

# processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


dataset = CLIPDataset(processor)

data_collator = DefaultDataCollator()

training_args = TrainingArguments(
    output_dir="clip_exp",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    # metric_for_best_model="accuracy",
    metric_for_best_model=None,
    push_to_hub=False
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    compute_metrics=None
)
trainer.train()