import torch
from PIL import Image
from dataset import SimpleDataset, make_dataset
from glob import glob
import os.path as osp
import evaluate
import numpy as np
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import DefaultDataCollator


def main():
    label2id, id2label = dict(), dict()

    IMAGE_ROOT_DIR = "/home/zjf/repos/proj_595/data/action_effect_image_rs"
    effect_dir_list = glob(osp.join(IMAGE_ROOT_DIR, '*'))
    labels = [osp.basename(effect_dir) for effect_dir in effect_dir_list]

    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
        
    from transformers import AutoImageProcessor
    checkpoint = "google/vit-base-patch16-224-in21k"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)

    from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    def transforms(examples):
        examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    ds_train, ds_val = make_dataset(transform=_transforms)
    
    
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)


    model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    )
    
    data_collator = DefaultDataCollator()


    training_args = TrainingArguments(
        output_dir="simple_exp",
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
        metric_for_best_model="accuracy",
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model('saved_model')

if __name__ == '__main__':
    main()

