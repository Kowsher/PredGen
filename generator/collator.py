import torch

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length

    def __call__(self, features):
        """
        Each feature is a dict like:
          {
              "input_left": "...",
              "input_right": "...",
              "label": ...
          }
        """
        # Extract left and right texts
        #print('features', features)
        left_texts = [f["input"] for f in features]
        right_texts = [f["output"] for f in features]
        labels = torch.tensor([f["labels"] for f in features])  # Ensure labels are tensors
    
    
        # Backup the original padding side
        original_padding_side = self.tokenizer.padding_side
        original_pad_token_id = self.tokenizer.pad_token_id
    
        # ------------------ LEFT side padding ------------------
        
        self.tokenizer.padding_side = "left"
        enc_left = self.tokenizer(
            left_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
    
        # ------------------ RIGHT side padding ------------------
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token_id = 0
        enc_right = self.tokenizer(
            right_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
    
        # Restore the original padding side (important if tokenizer is reused elsewhere)
        self.tokenizer.padding_side = original_padding_side
        self.tokenizer.pad_token_id = original_pad_token_id
    
        # Build the combined batch
        batch = {
            "input_ids": enc_left["input_ids"],
            "attention_mask": enc_left["attention_mask"],
            "output_ids": enc_right["input_ids"],
            "output_attention_mask": enc_right["attention_mask"],
            "labels": labels  # Include extracted labels here
        }
    
        return batch

class DataCollatorDynamicForTest:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length

    def __call__(self, features):
        """
        Each feature is a dict like:
          {
              "input_left": "...",
              "input_right": "...",
              "label": ...
          }
        """
        print('features', features)
        print('features.features', type(features[0]), features[0].keys())
  
        # Backup the original padding side
        original_padding_side = self.tokenizer.padding_side
        original_pad_token_id = self.tokenizer.pad_token_id
    
        # ------------------ LEFT side padding ------------------
        if "instruction" in list(features[0].keys()):
            left_texts = [f["input"] for f in features]
            self.tokenizer.padding_side = "left"
            enc_left = self.tokenizer(
                left_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            enc_left = {}
            enc_left["input_ids"] = None
            enc_left["attention_mask"] = None

        if "output" in list(features[0].keys()):
            self.tokenizer.padding_side = "right"
            #self.tokenizer.pad_token_id = 0
            right_texts = [f["output"] for f in features]
            enc_right = self.tokenizer(
                right_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            enc_right = {}
            enc_right["input_ids"] = None
            enc_right["attention_mask"] = None

        if "labels" in list(features[0].keys()):
            labels = torch.tensor([f["labels"] for f in features])
        else:
            labels = None
            
    

        self.tokenizer.padding_side = original_padding_side
        self.tokenizer.pad_token_id = original_pad_token_id
    
        # Build the combined batch
        batch = {
            "input_ids": enc_left["input_ids"],
            "attention_mask": enc_left["attention_mask"],
            "output_ids": enc_right["input_ids"],
            "output_attention_mask": enc_right["attention_mask"],
            "labels": labels  # Include extracted labels here
        }
    
        return batch
