from transformers import LayoutLMv2Processor
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image

processor = LayoutLMv2Processor\
    .from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

class ImageLayoutDataset(Dataset):
    def __init__(self, 
                 data,
                 processor = processor,
                 encode : bool = True) -> None:
        super().__init__()

        self.processor = processor

        if encode:
            self.X = []
            for example in tqdm(data):
                X= Df.encode(example)
                self.X.append(X)

        else:
            self.X = data

    def encode(
        self,
        example, 
    ):
        words = example['words']
        boxes = example['bboxes']
        image = Image.open(example['image_path'])
        word_labels = example['ner_tags']

        encoded_inputs = self.processor(
            image, 
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