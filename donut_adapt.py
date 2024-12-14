import os
import json
from pathlib import Path
import shutil
from datasets import load_dataset, load_from_disk
from transformers import (
    DonutProcessor, 
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from huggingface_hub import HfFolder, notebook_login
import random
from dotenv import load_dotenv

def prepare_dataset():
    """Prepare and process the SROIE dataset"""
    base_path = Path("data")
    metadata_path = base_path.joinpath("key")
    image_path = base_path.joinpath("img")
    metadata_list = []

    # Parse metadata
    for file_name in metadata_path.glob("*.json"):
        with open(file_name, "r") as json_file:
            data = json.load(json_file)
            text = json.dumps(data)
            if image_path.joinpath(f"{file_name.stem}.jpg").is_file():
                metadata_list.append({"text":text,"file_name":f"{file_name.stem}.jpg"})

    # Write jsonline file
    with open(image_path.joinpath('metadata.jsonl'), 'w') as outfile:
        for entry in metadata_list:
            json.dump(entry, outfile)
            outfile.write('\n')

    # Remove old metadata if it exists
    if metadata_path.exists():
        shutil.rmtree(metadata_path)
    
    # Load dataset
    dataset = load_dataset("imagefolder", data_dir=image_path, split="train")
    return dataset

def json2token(obj, new_special_tokens=[], update_special_tokens_for_json_key=True, sort_json_key=True):
    """Convert JSON object to token sequence"""
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys:
                if update_special_tokens_for_json_key:
                    new_special_tokens.append(fr"<s_{k}>") if fr"<s_{k}>" not in new_special_tokens else None
                    new_special_tokens.append(fr"</s_{k}>") if fr"</s_{k}>" not in new_special_tokens else None
                output += fr"<s_{k}>" + json2token(obj[k], new_special_tokens, update_special_tokens_for_json_key, sort_json_key) + fr"</s_{k}>"
            return output
    elif type(obj) == list:
        return r"<sep/>".join([json2token(item, new_special_tokens, update_special_tokens_for_json_key, sort_json_key) for item in obj])
    else:
        obj = str(obj)
        if f"<{obj}/>" in new_special_tokens:
            obj = f"<{obj}/>"
        return obj

def preprocess_documents(samples, task_start_token="<s>", eos_token="</s>"):
    """Preprocess documents for Donut model"""
    processed_images = []
    processed_texts = []

    for text_str, image in zip(samples["text"], samples["image"]):
        text_obj = json.loads(text_str)
        d_doc = task_start_token + json2token(text_obj) + eos_token
        image = image.convert('RGB')
        processed_images.append(image)
        processed_texts.append(d_doc)

    return {"image": processed_images, "text": processed_texts}

def transform_and_tokenize(sample, processor, split="train", max_length=512, ignore_id=-100):
    """Transform and tokenize the data"""
    try:
        pixel_values = processor(
            sample["image"], 
            random_padding=split == "train", 
            return_tensors="pt"
        ).pixel_values.squeeze()
    except Exception as e:
        print(f"Error: {e}")
        return {}

    input_ids = processor.tokenizer(
        sample["text"],
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"].squeeze(0)

    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = ignore_id
    return {"pixel_values": pixel_values, "labels": labels, "target_sequence": sample["text"]}

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get token from .env file
    token = os.getenv('HF_TOKEN')
    if not token:
        raise ValueError("Please ensure HF_TOKEN is set in your .env file")
    
    HfFolder.save_token(token)
    print("✓ Successfully set up Hugging Face token")
    
    # Prepare dataset
    print("\n📁 Preparing dataset...")
    dataset = prepare_dataset()
    print(f"✓ Dataset prepared with {len(dataset)} samples")
    
    # Initialize processor
    print("\n🔄 Initializing Donut processor...")
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    print("✓ Processor initialized")
    
    # Process dataset
    print("\n🔄 Processing documents...")
    proc_dataset = dataset.map(
        preprocess_documents,
        batched=True,
        batch_size=4
    )
    print("✓ Documents processed")
    
    # Add special tokens to tokenizer
    print("\n📝 Adding special tokens...")
    new_special_tokens = []
    processor.tokenizer.add_special_tokens({
        "additional_special_tokens": new_special_tokens + ["<s>", "</s>"]
    })
    print("✓ Special tokens added")
    
    # Update processor settings
    print("\n⚙️ Updating processor settings...")
    processor.feature_extractor.size = [720, 960]
    processor.feature_extractor.do_align_long_axis = False
    print("✓ Processor settings updated")
    
    # Transform and tokenize dataset
    print("\n🔄 Transforming and tokenizing dataset...")
    processed_dataset = proc_dataset.map(
        lambda x: transform_and_tokenize(x, processor),
        remove_columns=["image", "text"],
        batched=True,
        batch_size=4
    )
    print("✓ Dataset transformed and tokenized")
    
    # Split dataset
    print("\n📊 Splitting dataset...")
    processed_dataset = processed_dataset.train_test_split(test_size=0.1)
    print(f"✓ Dataset split into {len(processed_dataset['train'])} train and {len(processed_dataset['test'])} test samples")
    
    # Initialize model
    print("\n🔄 Initializing model...")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    print("✓ Model initialized")
    
    # Configure model
    print("\n⚙️ Configuring model...")
    model.config.encoder.image_size = (processor.feature_extractor.size['height'], 
                                     processor.feature_extractor.size['width'])
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s>'])[0]
    print("✓ Model configured")
    
    # Setup training arguments
    print("\n⚙️ Setting up training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir="donut-base-sroie",
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        weight_decay=0.01,
        fp16=True,
        logging_steps=50,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        predict_with_generate=True,
        report_to="tensorboard",
        hub_strategy="every_save",
        hub_model_id="YourHuggingFaceUsername/donut-base-sroie",  # Replace with your username
        push_to_hub=True,
        hub_token=HfFolder.get_token(),
    )
    print("✓ Training arguments set")
    
    # Initialize trainer
    print("\n🔄 Initializing trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
    )
    print("✓ Trainer initialized")
    
    # Train model
    print("\n🚀 Starting training...")
    trainer.train()
    print("✓ Training completed!")

if __name__ == "__main__":
    main()
