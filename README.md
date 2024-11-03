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
python src/BERT.py --dataset_path ./glue/mrpc_dataset --model_name bert-base-uncased
```

## Models  
A selection of fine-tuned models is available on my **Hugging Face** account. You can explore and use them at the following link:  
🔗 [https://huggingface.co/rambodazimi](https://huggingface.co/rambodazimi)


## Citation  
If you find this work helpful, please consider citing our paper:
```bibtex
```

## License  
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
