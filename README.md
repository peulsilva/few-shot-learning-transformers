# few-shot-learning-transformers

## Introduction

Few-shot learning in transformers refers to the ability of transformer-based models, such as BERT, GPT, or T5, to perform a task with only a small amount of training data. Traditionally, deep learning models require a large amount of labeled data to achieve high performance on a task. However, in many real-world scenarios, obtaining such data can be costly or time-consuming.

In few-shot learning, the model is trained on a dataset that contains examples of the task, but typically with very few instances per class or category. The model learns to generalize from this limited set of examples to perform the task accurately on unseen data. This is achieved through techniques like meta-learning, where the model learns how to learn from few examples, or through the use of pre-trained language models fine-tuned on a few examples for specific tasks.

Transformers are well-suited for few-shot learning due to their attention mechanism, which allows them to effectively capture complex patterns in the data and generalize from limited examples. By fine-tuning pre-trained transformer models on a few examples of a specific task, researchers and practitioners can achieve impressive results across various natural language processing tasks with minimal labeled data.

## Studied Approaches

### Text Classification
- Sentence Transformers Finetuning ([SetFit](https://arxiv.org/abs/2209.11055))
- Pattern Exploiting Training ([PET](https://arxiv.org/abs/2001.07676))

### Named Entity Recognition for Image Documents
- Pattern Exploiting Training ([PET](https://arxiv.org/abs/2001.07676))
- [CLASSBITE](https://arxiv.org/abs/2305.04928)
- [ContaiNER](https://arxiv.org/abs/2109.07589)


### Classification Utils
- [Focal Loss function for imbalanced datasets](https://arxiv.org/abs/1708.02002)
- Stratified train test split

## Usage
To install project dependencies, run:

```bash
pip install -r requirements.txt
```

## Leaderboard

### Few shot Named Entity Recognition for image documents

| Exemples         | Approche    | FUNSD | SROIE | CORD |
|------------------|-------------|-------|-------|------|
|                  | Finetuning  | 0.33  | 0.93  | 0.45 |
|       2          | PET         | 0.43  | 0.85  | N.A. |
|                  | Bio technique | 0.53  | 0.93  | N.A. |
|------------------|-------------|-------|-------|------|
|                  | Finetuning  | 0.41  | 0.94  | 0.56 |
|       5          | PET         | 0.48  | 0.92  | N.A. |
|                  | Bio technique | 0.53  | 0.96  | N.A. |
|------------------|-------------|-------|-------|------|
|                  | Finetuning  | 0.69  | 0.94  | 0.71 |
|       10         | PET         | 0.53  | 0.94  | N.A. |
|                  | Bio technique | 0.61  | 0.97  | N.A. |

### Few shot text classification

| Exemples         | Approche | AgNews | SST  |
|------------------|----------|--------|------|
|                  | Finetuning | 0.32   | 0.26 |
| 2                | SetFit     | 0.78   | 0.26 |
|                  | PET        | 0.76   | 0.30 |
|------------------|----------|--------|------|
|                  | Finetuning | 0.69   | 0.29 |
| 5                | SetFit     | 0.78   | 0.28 |
|                  | PET        | 0.81   | 0.32 |
|------------------|----------|--------|------|
|                  | Finetuning | 0.73   | 0.32 |
| 10               | SetFit     | 0.86   | 0.35 |
|                  | PET        | 0.84   | 0.35 |

### Few shot benchmark study

| Exemples         | Approche    | ADE Corpus | Tweet eval emotion | Wiki Q&A | Ethos |
|------------------|-------------|------------|--------------------|----------|-------|
|                  | Finetuning  | 0.43       | 0.44               | 0.04     | 0.50  |
| 50               | SetFit      | 0.30       | 0.42               | 0.09     | 0.56  |
|                  | PET         | 0.53       | 0.65               | 0.01     | 0.61  |
|------------------|-------------|------------|--------------------|----------|-------|
|                  | Finetuning  | 0.61       | 0.69               | 0.10     | 0.53  |
| 500              | SetFit      | 0.58       | 0.72               | 0.14     | 0.62  |
|                  | PET         | 0.76       | 0.73               | 0.13     | 0.70  |
