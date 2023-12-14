from typing import List, Dict
from datasets import Dataset
from transformers import AutoTokenizer
from PIL import Image

def unnormalize_bbox(bbox, width, height):
    return [
        bbox[0] * width /1000,
        bbox[1] * height /1000,
        bbox[2] * width /1000,
        int((bbox[3] * height / 1000)),
    ]


def get_unnormalized_bboxes(dataset):
    unnormalized_bbox = []

    for document in dataset:
        document_boxes = []
        img_pil = Image.open(document['image_path'])
        
        for box in document['bboxes']:
            new_box = unnormalize_bbox(box,img_pil.width, img_pil.height)
            document_boxes.append(new_box)

        unnormalized_bbox.append(document_boxes)

    return unnormalized_bbox

def generate_dataset(
    dataset : Dataset,
    label_names : List[str],
    idx2label : Dict[int, str],
    tokenizer : AutoTokenizer,
    n_shots : int,
):
    new_labels = []
    new_words = []
    new_bboxes = []
    for document_id, document in enumerate(dataset):
        if document_id >= n_shots:
            break
        
        for label in label_names:
            new_labels_document = [1,-100]
            new_words_document = [label, tokenizer.sep_token]
            new_bboxes_document = [[0,0,0,0], [0,0,0,0]]

            for idx, word in enumerate(document['words']):
                this_label = document['ner_tags'][idx]
                bboxes = document['bboxes'][idx]

                binary_label = int(idx2label[this_label] == label)

                new_labels_document.append(binary_label)
                new_bboxes_document.append(bboxes)
                new_words_document.append(word)
            
            new_labels.append(new_labels_document)
            new_words.append(new_words_document)
            new_bboxes.append(new_bboxes_document)


    return Dataset.from_dict(
            {
                "ner_tags": new_labels,
                "words": new_words,
                "bboxes" : new_bboxes
            },
        )

def generate_text_from_same_line(
    dataset : Dataset,
    unnormalized_bbox : List[List[int]],
    n_shots : int
):
    
    new_words = []
    new_boxes = []
    new_tags = []
    for document_idx in range(len(dataset['words'])):
        if document_idx >= n_shots: 
            break
        
        document = dataset[document_idx]

        lines = {}

        for idx, token in enumerate(document['words']):
            label = document['ner_tags'][idx]
            bbox = document['bboxes'][idx]
            y_coordinate = unnormalized_bbox[document_idx][idx][3]
            x_coordinate = unnormalized_bbox[document_idx][idx][0]

            is_added = False
            for line, line_text in lines.items():
                if abs(y_coordinate - line <=1):
                    line_text.append([x_coordinate, token, label, bbox])
                    is_added = True
                    continue
                
            if not is_added:
                lines[y_coordinate] = [[x_coordinate, token, label, bbox]]

            for k, v in lines.items():
                lines[k] = sorted(v, key = lambda x: x[0])

        for k, v in lines.items():
            line_words = []
            line_boxes = []
            line_tags = []
            
            for _,t, l, b in v:
                line_words.append(t)
                line_tags.append(l)
                line_boxes.append(b)

            new_words.append(line_words)
            new_tags.append(line_tags)
            new_boxes.append(line_boxes)

    return Dataset.from_dict(
            {
                "ner_tags": new_tags,
                "words": new_words,
                "bboxes" : new_boxes
            },
        )