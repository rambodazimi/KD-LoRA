# KD-LoRA

This repository provides the official implementation for the paper:  
**KD-LoRA: A Hybrid Approach to Efficient Fine-Tuning with LoRA and Knowledge Distillation**

## Overview  
KD-LoRA combines **Low-Rank Adaptation (LoRA)** and **Knowledge Distillation** to enable lightweight, effective, and efficient fine-tuning of large language models.

## Authors  

| Name            | Email Address              |
|-----------------|----------------------------|
| Rambod Azimi    | rambod.azimi@mail.mcgill.ca     |
| Rishav Rishav   | mail.rishav9@gmail.com     |
| Marek Teichmann   | marek@cm-labs.com     |
| Samira Ebrahimi Kahou  | samira.ebrahimi.kahou@gmail.com     |


## Installation  
```bash
# Clone the repository
git clone https://github.com/rambodazimi/kd-lora.git
cd kd-lora

# Install dependencies
pip install -r requirements.txt
```

## Usage  
Instructions for running experiments and fine-tuning models will go here.
```bash
python src/BERT.py --dataset_path <DATASET_PATH> --model_name <MODEL_NAME> --learning_rate <LEARNING_RATE> --train_batch_size <TRAIN_BATCH_SIZE> --eval_batch_size <EVAL_BATCH_SIZE> --num_train_epochs <NUM_TRAIN_EPOCHS> --weight_decay <WEIGHT_DECAY>
```

### Arguments

- **--dataset_path** (str): Path to the dataset. Example options for GLUE tasks include:
  - `./glue/mrpc_dataset` for the MRPC task (default)
  - `./glue/cola_dataset` for the CoLA task
  - `./glue/sst2_dataset` for the SST-2 task
  - `./glue/wnli_dataset` for the WNLI task
  - `./glue/rte_dataset` for the RTE task
  - `./glue/qqp_dataset` for the QQP task
  - `./glue/stsb_dataset` for the STS-B task
  - `./glue/qnli_dataset` for the QNLI task
  - `./glue/mnli_dataset` for the MNLI task
  
- **--model_name** (str): Model name or path to a pre-trained model. For example:
  - `bert-base-uncased` (default)
  - You can also specify a local path to a model if it's saved on your machine.

- **--learning_rate** (float): Learning rate for the training process. Default is `5e-5`.

- **--train_batch_size** (int): Training batch size per device. Default is `32`.

- **--eval_batch_size** (int): Evaluation batch size per device. Default is `32`.

- **--num_train_epochs** (int): Number of epochs for training. Default is `3`.

- **--weight_decay** (float): Weight decay rate. Default is `0.01`.

### Example Commands

1. **Fine-tune BERT on the MRPC task (default settings)**

    ```bash
    python src/BERT.py --dataset_path ./glue/mrpc_dataset --model_name bert-base-uncased
    ```

2. **Fine-tune BERT on the SST-2 task with a custom learning rate and batch size**

    ```bash
    python src/BERT.py --dataset_path ./glue/sst2_dataset --model_name bert-base-uncased --learning_rate 3e-5 --train_batch_size 16 --eval_batch_size 16
    ```

3. **Fine-tune BERT on the MNLI task with a specific number of epochs and weight decay**

    ```bash
    python src/BERT.py --dataset_path ./glue/mnli_dataset --model_name bert-base-uncased --num_train_epochs 5 --weight_decay 0.02
    ```

4. **Fine-tune a custom model on the QNLI task**

    ```bash
    python src/BERT.py --dataset_path ./glue/qnli_dataset --model_name <path/to/custom_model> --learning_rate 2e-5
    ```

This should cover various scenarios for fine-tuning the BERT model on different GLUE tasks using the provided arguments. Adjust the arguments as needed to customize the training process for specific tasks or datasets.

## Models  
A selection of fine-tuned models is available on my **Hugging Face** account. You can explore and use them at the following link:  
🔗 [https://huggingface.co/rambodazimi](https://huggingface.co/rambodazimi)


## Citation  
If you find this work helpful, please consider citing our paper:
```bibtex
```

## License  
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
