import argparse
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
import torch.nn.functional as F


def main(args):
    # Step 1: Fine-tune a Teacher Model
    print(f"Fine-tuning the teacher model: {args.teacher_model_name}")
    teacher_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_model_name, num_labels=args.num_labels)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name)

    teacher_training_args = TrainingArguments(
        output_dir="./teacher_results",
        learning_rate=args.teacher_learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
    )

    teacher_dataset = load_from_disk(args.dataset_path)
    tokenized_teacher_dataset = teacher_dataset.map(
        lambda x: teacher_tokenizer(x["sentence1"], x["sentence2"], padding="max_length", truncation=True),
        batched=True
    )

    # Define trainer for teacher model
    teacher_trainer = Trainer(
        model=teacher_model,
        args=teacher_training_args,
        train_dataset=tokenized_teacher_dataset["train"],
        eval_dataset=tokenized_teacher_dataset["validation"]
    )

    teacher_trainer.train()

    # Save teacher model predictions (logits) as soft labels
    teacher_logits = teacher_trainer.predict(tokenized_teacher_dataset["train"]).predictions
    teacher_soft_labels = torch.tensor(teacher_logits)

    # Step 2: Initialize a Smaller Student Model with LoRA
    print(f"Initializing student model: {args.student_model_name} with LoRA")
    student_model = AutoModelForSequenceClassification.from_pretrained(args.student_model_name, num_labels=args.num_labels)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model_name)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS"
    )

    # Apply LoRA configuration to the student model
    student_model = get_peft_model(student_model, lora_config)

    # Freeze all layers except LoRA parameters
    for param in student_model.parameters():
        param.requires_grad = False
    for name, param in student_model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True  # Only LoRA weights are trainable

    # Step 3: Distillation from Teacher to Student
    print("Starting knowledge distillation from teacher to student")
    student_training_args = TrainingArguments(
        output_dir="./student_results",
        learning_rate=args.student_learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
    )

    def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
        # Compute the distillation loss with temperature scaling
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean"
        ) * (temperature ** 2)
        hard_loss = F.cross_entropy(student_logits, labels)
        return alpha * soft_loss + (1 - alpha) * hard_loss

    # Define a custom training loop for distillation
    class DistillationTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            student_logits = outputs.logits
            teacher_logits = teacher_soft_labels[inputs["input_ids"].shape[0]]  # Align teacher logits with batch size
            loss = distillation_loss(student_logits, teacher_logits, labels)
            return (loss, outputs) if return_outputs else loss

    # Tokenize student dataset
    tokenized_student_dataset = teacher_dataset.map(
        lambda x: student_tokenizer(x["sentence1"], x["sentence2"], padding="max_length", truncation=True),
        batched=True
    )

    # Initialize Distillation Trainer
    student_trainer = DistillationTrainer(
        model=student_model,
        args=student_training_args,
        train_dataset=tokenized_student_dataset["train"],
        eval_dataset=tokenized_student_dataset["validation"]
    )

    # Train student model with knowledge distillation
    student_trainer.train()

    # Evaluate student model
    student_trainer.evaluate()

    # Save the fine-tuned LoRA student model
    output_dir = "./fine_tuned_student_model"
    student_model.save_pretrained(output_dir)
    student_tokenizer.save_pretrained(output_dir)
    print(f"Student model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation with LoRA-enhanced Student Model")

    # Model arguments
    parser.add_argument("--teacher_model_name", type=str, default="bert-large-uncased", help="Name of the teacher model")
    parser.add_argument("--student_model_name", type=str, default="bert-base-uncased", help="Name of the student model")

    # Dataset and training parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels for the classification task")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")

    # LoRA parameters
    parser.add_argument("--rank", type=int, default=8, help="Rank of LoRA matrices")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate for LoRA layers")

    # Learning rates for teacher and student
    parser.add_argument("--teacher_learning_rate", type=float, default=5e-5, help="Learning rate for the teacher model")
    parser.add_argument("--student_learning_rate", type=float, default=5e-5, help="Learning rate for the student model")

    args = parser.parse_args()
    main(args)