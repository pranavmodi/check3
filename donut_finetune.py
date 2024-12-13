import os
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Any

import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from huggingface_hub import HfFolder, notebook_login

class DonutFineTuner:
    def __init__(self):
        self.new_special_tokens = []
        self.task_start_token = "<s>"
        self.eos_token = "</s>"

    def prepare_dataset(self, base_path: str) -> None:
        """Prepare the dataset by organizing metadata and images"""
        base_path = Path(base_path)
        metadata_path = base_path.joinpath("key")
        image_path = base_path.joinpath("img")
        metadata_list = []

        # Parse metadata
        for file_name in metadata_path.glob("*.json"):
            with open(file_name, "r") as json_file:
                data = json.load(json_file)
                text = json.dumps(data)
                if image_path.joinpath(f"{file_name.stem}.jpg").is_file():
                    metadata_list.append({
                        "text": text,
                        "file_name": f"{file_name.stem}.jpg"
                    })

        # Write jsonline file
        with open(image_path.joinpath('metadata.jsonl'), 'w') as outfile:
            for entry in metadata_list:
                json.dump(entry, outfile)
                outfile.write('\n')

        # Remove old metadata
        shutil.rmtree(metadata_path)

    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, 
                  sort_json_key: bool = True) -> str:
        """Convert an ordered JSON object into a token sequence"""
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            
            output = ""
            keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
            
            for k in keys:
                if update_special_tokens_for_json_key:
                    start_token = fr"<s_{k}>"
                    end_token = fr"</s_{k}>"
                    if start_token not in self.new_special_tokens:
                        self.new_special_tokens.append(start_token)
                    if end_token not in self.new_special_tokens:
                        self.new_special_tokens.append(end_token)
                        
                output += (
                    fr"<s_{k}>"
                    + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                    + fr"</s_{k}>"
                )
            return output
            
        elif isinstance(obj, list):
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) 
                 for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in self.new_special_tokens:
                obj = f"<{obj}/>"
            return obj

    def preprocess_documents(self, samples: Dict) -> Dict:
        """Preprocess documents for the Donut model"""
        processed_images = []
        processed_texts = []

        for text_str, image in zip(samples["text"], samples["image"]):
            text_obj = json.loads(text_str)
            d_doc = self.task_start_token + self.json2token(text_obj) + self.eos_token
            image = image.convert('RGB')

            processed_images.append(image)
            processed_texts.append(d_doc)

        return {"image": processed_images, "text": processed_texts}

    def transform_and_tokenize(self, sample: Dict, split: str = "train", 
                             max_length: int = 512, ignore_id: int = -100) -> Dict:
        """Transform and tokenize the data"""
        try:
            pixel_values = self.processor(
                sample["image"], 
                random_padding=split == "train", 
                return_tensors="pt"
            ).pixel_values.squeeze()
        except Exception as e:
            print(f"Error processing image: {e}")
            return {}

        input_ids = self.processor.tokenizer(
            sample["text"],
            add_special_tokens=False,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = ignore_id
        
        return {
            "pixel_values": pixel_values, 
            "labels": labels, 
            "target_sequence": sample["text"]
        }

    def setup_model_and_processor(self) -> None:
        """Setup the Donut model and processor"""
        self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        self.processor.tokenizer.add_special_tokens({
            "additional_special_tokens": self.new_special_tokens + 
                                      [self.task_start_token] + 
                                      [self.eos_token]
        })
        
        # Update processor settings
        self.processor.feature_extractor.size = {'height': 720, 'width': 960}
        self.processor.feature_extractor.do_align_long_axis = False

        # Setup model
        self.model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
        new_emb = self.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))
        
        # Configure model settings
        self.model.config.encoder.image_size = (
            self.processor.feature_extractor.size['height'],
            self.processor.feature_extractor.size['width']
        )
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.decoder_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(['<s>'])[0]

    def train(self, processed_dataset: Dict) -> None:
        """Train the model"""
        training_args = Seq2SeqTrainingArguments(
            output_dir="donut-base-sroie",
            num_train_epochs=3,
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            weight_decay=0.01,
            fp16=True,
            logging_steps=100,
            save_total_limit=2,
            evaluation_strategy="no",
            save_strategy="epoch",
            predict_with_generate=True,
            report_to="tensorboard",
            push_to_hub=True,
            hub_strategy="every_save",
            hub_model_id="donut-base-sroie",
            hub_token=HfFolder.get_token(),
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_dataset["train"],
        )

        trainer.train()

def main():
    # Login to Hugging Face
    notebook_login()
    
    # Initialize DonutFineTuner
    donut = DonutFineTuner()
    
    # Prepare dataset
    donut.prepare_dataset("data")
    
    # Load dataset
    dataset = load_dataset("imagefolder", data_dir="data/img", split="train")
    
    # Preprocess dataset
    proc_dataset = dataset.map(
        donut.preprocess_documents,
        batched=True,
        batch_size=4
    )
    
    # Transform and tokenize
    processed_dataset = proc_dataset.map(
        donut.transform_and_tokenize,
        remove_columns=["image", "text"],
        batched=True,
        batch_size=4
    )
    
    # Split dataset
    processed_dataset = processed_dataset.train_test_split(test_size=0.1)
    
    # Setup model and processor
    donut.setup_model_and_processor()
    
    # Train model
    donut.train(processed_dataset)

if __name__ == "__main__":
    main()

