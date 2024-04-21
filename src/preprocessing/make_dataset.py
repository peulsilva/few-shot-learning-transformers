from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict
import torch
from copy import copy


class ImageLayoutDataset(Dataset):
    def __init__(self, 
                 data,
                 tokenizer,
                 device : str = 'cuda',
                 encode : bool = True,
                 tokenize_all_labels : bool = False,
                 valid_labels_keymap : Dict = None) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.device = device
        self.valid_labels_keymap = valid_labels_keymap
        self.tokenize_all_labels = tokenize_all_labels

        if encode:
            self.X = []
            for example in tqdm(data):
                X= self.encode(example)
                self.X.append(X)

        else:
            self.X = data
    
    def tokenize_labels(
        self,
        ner_tags : List,
        tokens 
    )-> torch.Tensor:
        """Aligns and tokenize labels

        Args:
            ner_tags (List): labels
            tokens (_type_): tokens

        Returns:
            torch.Tensor: tokenized labels
        """        
        
        labels = []

        word_ids = tokens.word_ids()  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                if self.valid_labels_keymap is not None:
                    label_ids.append(self.valid_labels_keymap[ner_tags[word_idx]])
                else:
                    label_ids.append(ner_tags[word_idx])
            else:
                if self.tokenize_all_labels:
                    if self.valid_labels_keymap is not None:
                        label_ids.append(self.valid_labels_keymap[ner_tags[word_idx]])
                    else:
                        label_ids.append(ner_tags[word_idx])
                else: 
                    label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

        return torch.Tensor(labels).to(torch.int64)
    
    
    def tokenize_boxes(
        self,
        words : List,
        boxes : List,
    ):
        
        token_boxes = []
        max_seq_length = 512
        pad_token_box = [0,0,0,0]
        
        for word, box in zip(words, boxes):
            word_tokens = self.tokenizer.tokenize(word)
            token_boxes.extend([box] * len(word_tokens))

        # Truncation of token_boxes
        special_tokens_count = 2 
        if len(token_boxes) > max_seq_length - special_tokens_count:
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]

        # add bounding boxes of cls + sep tokens
        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        # Padding of token_boxes up the bounding boxes to the sequence length.
        input_ids = self.tokenizer(' '.join(words), truncation=True)["input_ids"]
        padding_length = max_seq_length - len(input_ids)
        token_boxes += [pad_token_box] * padding_length

        return torch.tensor(token_boxes)

    def encode(
        self,
        example, 
    ):
        words = example['words']
        boxes = example['bboxes']
        # image = Image.open(example['image_path'])s
        word_labels = example['ner_tags']

        
        tokens = self.tokenizer(
            words, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt",
            is_split_into_words= True
        )

        labels = self.tokenize_labels(word_labels,tokens)
        bbox = self.tokenize_boxes(words, boxes)

        tokens = {
            **tokens,
            "labels": labels,
            "bbox": bbox
        }

        for (k,v ) in tokens.items():
            tokens[k] = v.to(self.device)
    
        return tokens

        
    
    def __getitem__(self, index: int):
        return self.X[index]

    def __len__(self):
        return len(self.X)
    

class PatternExploitingDataset(Dataset):
    def __init__(self, 
                 data,
                 tokenizer,
                 pattern_fn : callable,
                 num_samples : int = 10) -> None:
        super().__init__()
        
        self.label_names = data\
            .features['ner_tags']\
            .feature\
            .names
        
        self.label_keymap = {k:v for k,v in enumerate(self.label_names)} 

        self.pattern = pattern_fn
        self.tokenizer = tokenizer

        self.num_samples = num_samples
        self._ignore(data[0:num_samples])

        # self.boxes = self.tokenize_boxes(self.words, self.boxes)


    
    def _ignore(
        self, 
        data,
        should_ignore = [";", ":", ".", " ", ""]
    ):
        self.boxes = []
        self.words = []
        self.labels = []
        for i in range(self.num_samples):
            words = data['words'][i]
            labels = data['ner_tags'][i]
            boxes = data['bboxes'][i]

            this_words = []
            this_labels = []
            this_boxes = []

            for j in range(len(words)):

                if not words[j] in should_ignore:
                    this_words.append(words[j])
                    this_labels.append(labels[j])
                    this_boxes.append(boxes[j])

            self.words.append(this_words)
            self.labels.append(this_labels)
            self.boxes.append(this_boxes)
    
    def __getitem__(self, document_index : int):
        phrases = []
        targets = []
        boxes = []

        words = self.words[document_index]
        labels = self.labels[document_index]

        for idx, word in enumerate(words):
            label_idx = labels[idx]
            label_name = self.label_keymap[label_idx]

            phrase = self.pattern(word, self.tokenizer)
            phrases.append(phrase)

            if label_name == "O":
                targets.append("NOTHING") 
            else :
                targets.append(label_name[2:])

        return phrases, targets, 
    
    def __len__(self):
        return len(self.words)
                

class SplitWordsDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        pattern_fn : callable,
        separators = [".", ":", "?"],
        label_names = None
    ) -> None:
        super().__init__()

        self.tokenizer= tokenizer
        self.separators = separators

        self.pattern_fn = pattern_fn

        if label_names == None:
            self.label_names : List[str] = data\
                .features['ner_tags']\
                .feature\
                .names

        else:
            self.label_names = label_names
        self.label_keymap = {k:v for k,v in enumerate(self.label_names)} 

        self.raw_data = data

        self.process(data)

    def process(self, data):
        self.processed_data = []

        for idx in tqdm(range(len(data))):
            example = data[idx]
            words = example['words']

            full_text = ' '.join(words)
            new_full_text = copy(full_text)

            for sep in self.separators:
                new_full_text = new_full_text.replace(sep, ".")

            split = SplitWordsDataset.split_string(
                new_full_text, 
                split_char=".", 
                min_words=3    
            )
            
            patterns = self.create_pattern(
                split,
                example['ner_tags'],
                words,
            )

            self.processed_data.append(patterns)

    @staticmethod
    def split_string(
        input_string: str, 
        split_char: str, 
        min_words: int = 3,
    ):
        words = input_string.split(split_char)
        result = []
        current_split = []

        for word in words:
            # Check if adding the current word would exceed the minimum word count
            if len(current_split) + 1 <= min_words:
                current_split.append(word)
            else:
                # If adding the word exceeds the minimum count, start a new split
                result.append(split_char.join(current_split))
                current_split = [word]

        # Add the remaining words to the result
        if current_split:
            result.append(split_char.join(current_split))

        return result
    
    def create_pattern(
        self,
        split: List,
        targets : List,
        words,
    ):
        pattern_list = []
        idx = 0
        for phrase in split:
            for word in phrase.split(" "):

                if len(word) < 2:
                    continue
                
                while word.split('.')[0] not in words[idx] :
                    idx += 1
                    
                label = targets[idx]
                real_name_label = self.label_keymap[label]
        
                if real_name_label == "O":
                    # continue
                    real_name_label= "none" 
                else :
                    real_name_label = real_name_label[2:].lower()

                pattern = self.pattern_fn(phrase, word, self.tokenizer)
                pattern_list.append({"pattern": pattern,
                                    "label": real_name_label})
        
        return pattern_list
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, index) :
        return self.processed_data[index]
        
                
    

