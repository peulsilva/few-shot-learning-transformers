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

| Exemples         | Approche    | FUNSD | SROIE | CORD |
|------------------|-------------|-------|-------|------|
|                  | Finetuning  | 0.33  | 0.93  | 0.45 |
| $|\mathcal{T}| = 2$ | PET         | 0.43  | 0.85  | N.A. |
|                  | Bio technique | 0.53  | 0.93  | N.A. |
|------------------|-------------|-------|-------|------|
|                  | Finetuning  | 0.41  | 0.94  | 0.56 |
| $|\mathcal{T}| = 5$ | PET         | 0.48  | 0.92  | N.A. |
|                  | Bio technique | 0.53  | 0.96  | N.A. |
|------------------|-------------|-------|-------|------|
|                  | Finetuning  | 0.69  | 0.94  | 0.71 |
| $|\mathcal{T}| = 10$| PET         | 0.53  | 0.94  | N.A. |
|                  | Bio technique | 0.61  | 0.97  | N.A. |

