from transformers import LayoutLMv2Processor, LayoutLMv2Tokenizer
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image

tokenizer = LayoutLMv2Tokenizer\
    .from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

class ImageLayoutDataset(Dataset):
    def __init__(self, 
                 data,
                 tokenizer = tokenizer,
                 encode : bool = True) -> None:
        super().__init__()

        self.tokenizer = tokenizer

        if encode:
            self.X = []
            for example in tqdm(data):
                X= self.encode(example)
                self.X.append(X)

        else:
            self.X = data

    def encode(
        self,
        example, 
    ):
        words = example['words']
        boxes = example['bboxes']
        # image = Image.open(example['image_path'])s
        word_labels = example['ner_tags']

        
        encoded_inputs = self.tokenizer(
            # image, 
            words, 
            boxes=boxes, 
            word_labels=word_labels, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
    
        return encoded_inputs

        

        
    
    def __getitem__(self, index: int):
        return self.X[index]

    def __len__(self):
        return len(self.X)