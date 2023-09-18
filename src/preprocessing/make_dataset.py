import numpy as np

def count_labels(dataset):
    unique_labels = []
    for img_data in dataset['train']['ner_tags']:
        unique_labels.append(np.unique(img_data))
    unique_labels = np.unique(np.concatenate(unique_labels))
    return unique_labels.size
    