from torch.utils.data import Dataset
from typing import List
from tqdm import tqdm

class LaserProcessor(Dataset):
    def __init__(self, 
                 data : List,
                 tokenizer,
                 device : str = "cuda") -> None:
        super().__init__()

        self.label_names = data\
            .features['ner_tags']\
            .feature\
            .names
        
        self.label_keymap = {k:v for k,v in enumerate(self.label_names)} 
        self.device = device
        self.tokenizer = tokenizer

        self.X = []
        self.y = []

        self.__process(data)

    def __process(self, data):
        for j in tqdm(range(len(data))):
            input_phrase =  " ".join(data[j]['words'])
            target_phrase = []
            n_words_in_doc = len(data[j]['words'])

            label_i_1 = None
            for idx in range(n_words_in_doc):
                word_i = data[j]['words'][idx]
                label_i = self.label_keymap[
                    data[j]['ner_tags'][idx]
                ]


                # has_tag = label_i[0] != 'O'
                is_begining = label_i[0] == 'B' \
                    or ((label_i == "O" ) and (label_i_1 is not None) and (label_i_1 != label_i) ) \
                    or idx == 0
                
                is_end = False
                if idx >= n_words_in_doc - 1:
                    is_end = True

                elif self.label_keymap[data[j]['ner_tags'][idx+1]] != label_i:
                    is_end = True 
                    next_label = self.label_keymap[data[j]['ner_tags'][idx+1]]

                    if next_label != "O" and label_i != "O":
                        if not next_label[0] == "B":
                            if next_label.split('-')[1] == label_i.split('-')[1]:
                                is_end = False

                
                elif is_begining and label_i != "O": 
                    if self.label_keymap[data[j]['ner_tags'][idx+1]] == label_i:
                        is_end = True


                # if has_tag:
                if is_begining:
                    target_phrase.append('[B]')
                

                target_phrase.append(word_i)

                if is_end:
                    target_phrase.append("[E]")
                    if label_i == "O":
                        target_phrase.append("NONE")
                    else:
                        target_phrase.append(label_i.split('-')[1])
                    target_phrase.append("[T]")

                label_i_1 = label_i
                # else:
                #     target_phrase.append(word_i)

            for idx in range(len(target_phrase)):
                if idx == 0:
                    continue
                
                word_i_1 = target_phrase[idx - 1]
                word_i = target_phrase[idx]

                if word_i_1 == "[T]" and word_i != "[B]":
                    target_phrase.insert(idx, "[B]")

            self.X.append([input_phrase, " ".join(target_phrase)])
            # self.y.append()

    def convert_text_to_labels(self, text : List[str]):
        mapping = {
            "NONE": 0,
            "QUESTION": 1,
            "ANSWER": 2,
            "HEADER": 3
        }

        set_label_names = set(["NONE", "QUESTION", "ANSWER", "HEADER", "[B]", "[E]", "[T]"])
        labels = []
        n = len(text)
        idx = 0
        set_j = set()
        while idx < n:
            word = text[idx]
            
            if word == "[T]":
                tag_name = text[idx - 1]
                # print(true_phrase[idx - 3], true_phrase[idx - 2], true_phrase[idx - 1])
                j = idx - 3
                while j > 0 and text[j] != "[B]":
                    j -= 1
                
                while j <= idx -3:

                    set_j.add(j)
                    if not text[j] in set_label_names:
                        labels.append(mapping[tag_name])
                    j+=1
                    
            idx +=1
        
        return labels

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index]