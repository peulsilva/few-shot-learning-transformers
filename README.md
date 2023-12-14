# few-shot-learning-transformers

This repository contains code and resources for training and using Transformers-based models for document layout analysis and information extraction on three different datasets: FUNSD, SROIE, and CORD. We leverage popular Transformer architectures like BERT and LayoutLM to tackle document understanding tasks.

## :closed_book: Introduction
Document layout analysis and information extraction are essential tasks in the field of Natural Language Processing (NLP) and Document Understanding. This project aims to provide state-of-the-art solutions for these tasks by using Transformer-based models.

## :pencil: Key Features
* Fine-tuned BERT models for Named Entity Recognition (NER) on FUNSD and CORD datasets.
* Fine-tuned LayoutLM models for structured data extraction from invoices using SROIE dataset.
* Easily customizable and extensible codebase for training on other document understanding tasks and datasets.
* Evaluation metrics and visualization tools for assessing model performance.

## :book: Models
We have fine-tuned LayoutLM models for information extraction on the CORD dataset. 

## :computer: Usage
To install project dependencies, run:

```bash
pip install -r requirements.txt
```

## Leaderboard

| number of shots | Dataset | Technique | Performance (F1 Score) | 
| - | - | - | - | 
| 2 | FUNSD | BioTechnique | 0.53 |
| 2 | FUNSD | PET (DistilBERT) | 0.43 |
| 2 | FUNSD | Finetuning (LayoutLM) | 0.3 |
| 5 | FUNSD | BioTechnique (BERT) | 0.53 |
| 5 | FUNSD | PET (DistilBERT) | 0.48 |
| 5 | FUNSD | Finetuning (LayoutLM) | 0.41 |
| 10 | FUNSD | Finetuning (LayoutLM) | 0.69 |
| 10 | FUNSD | BioTechnique (BERT) | 0.61 |
| 10 | FUNSD | PET (DistilBERT) | 0.53 |

