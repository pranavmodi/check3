import os
import json
from pathlib import Path
import shutil
from datasets import load_dataset
from transformers import (
    DonutProcessor, 
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from huggingface_hub import HfFolder
import random
from dotenv import load_dotenv
from PIL import Image

def gather_json_keys(json_obj, keys_set=None):
    """Recursively gather all keys from the JSON structure."""
    if keys_set is None:
        keys_set = set()
    if isinstance(json_obj, dict):
        for k, v in json_obj.items():
            keys_set.add(k)
            gather_json_keys(v, keys_set)
    elif isinstance(json_obj, list):
        for item in json_obj:
            gather_json_keys(item, keys_set)
    return keys_set

def prepare_dataset():
    """Prepare and process the SROIE dataset by creating a metadata.jsonl 
       and integrating it with the dataset loaded via imagefolder."""
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
                metadata_list.append({"text": text, "file_name": f"{file_name.stem}.jpg"})

    # Write jsonline file
    metadata_file = image_path.joinpath('metadata.jsonl')
    with open(metadata_file, 'w') as outfile:
        for entry in metadata_list:
            json.dump(entry, outfile)
            outfile.write('\n')

    # Remove old metadata directory if it exists
    if metadata_path.exists():
        shutil.rmtree(metadata_path)

    # Now load the dataset from imagefolder
    # This will give a dataset with "image" and a "label" column (if any).
    dataset = load_dataset("imagefolder", data_dir=image_path, split="train")

    # The dataset from imagefolder won't have the 'text' field. We must integrate it.
    # We'll create a dictionary from file_name to text for quick lookup.
    text_map = {item["file_name"]: item["text"] for item in metadata_list}

    def add_text(example):
        # The 'imagefolder' dataset typically provides 'image' and possibly 'label'
        # Let's assume it has a 'image' PIL object and a filename in 'image.filename'
        # If not, we need to rely on a known attribute: 'image' column returns PIL image, 
        # but no filename by default. We'll rely on the dataset builder's attributes.
        
        # As of now, datasets.Image feature doesn't store filename, so we need a heuristic:
        # The order of images in imagefolder is typically alphabetical by filename.
        # If filenames are consistent, we can rely on 'id' index and rebuild the name:
        # But better: we can load dataset with 'imagefolder' which provides a 'path' column if we specify `keep_filename=True`
        # If that's not possible, we must guess. Let's assume we re-load with:
        #   dataset = load_dataset("imagefolder", data_dir=image_path, split="train", keep_in_memory=True)
        # and then 'example["image"]' might include 'path'.
        
        # If not available, let's add a pre-step: The images are sorted by `imagefolder` in alphabetical order.
        # We'll rely on dataset.features to see if 'image' is a dict with 'path'.
        
        # Let's print features to confirm. We'll assume path is available:
        # For safety, let's do:
        path = example["image"].filename
        if path:
            fname = os.path.basename(path)
            if fname in text_map:
                example["text"] = text_map[fname]
            else:
                # If no match found, fallback to empty text or raise warning
                example["text"] = json.dumps({})
        else:
            # If no path is available, we need another strategy. Let's just put empty text.
            example["text"] = json.dumps({})
        return example

    dataset = dataset.map(add_text)

    return dataset

def json2token(obj, sort_json_key=True):
    """Convert JSON object to token sequence without dynamic token addition."""
    if isinstance(obj, dict):
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            keys = sorted(obj.keys()) if sort_json_key else obj.keys()
            for k in keys:
                output += f"<s_{k}>" + json2token(obj[k], sort_json_key=sort_json_key) + f"</s_{k}>"
            return output
    elif isinstance(obj, list):
        return "<sep/>".join([json2token(item, sort_json_key=sort_json_key) for item in obj])
    else:
        return str(obj)

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
            random_padding=(split == "train"),
            return_tensors="pt"
        ).pixel_values.squeeze(0)
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
    token = os.getenv('HF_TOKEN')
    if not token:
        raise ValueError("Please ensure HF_TOKEN is set in your .env file")

    HfFolder.save_token(token)
    print("✓ Successfully set up Hugging Face token")

    # Prepare dataset
    print("\n📁 Preparing dataset...")
    dataset = prepare_dataset()
    print(f"✓ Dataset prepared with {len(dataset)} samples")

    # Gather all possible keys from the dataset's JSON texts for special tokens
    all_keys = set()
    for text_str in dataset["text"]:
        data = json.loads(text_str)
        all_keys.update(gather_json_keys(data))

    # Build special tokens for keys
    new_special_tokens = ["<s>", "</s>"]  # basic tokens
    for k in all_keys:
        new_special_tokens.append(f"<s_{k}>")
        new_special_tokens.append(f"</s_{k}>")
    new_special_tokens.append("<sep/>")

    # Initialize processor
    print("\n🔄 Initializing Donut processor...")
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    processor.tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
    print("✓ Processor initialized and tokens added")

    # Process dataset
    print("\n🔄 Processing documents...")
    proc_dataset = dataset.map(
        preprocess_documents,
        batched=True,
        batch_size=4
    )
    print("✓ Documents processed")

    # Update processor settings
    print("\n⚙️ Updating processor settings...")
    # Set the size as a dict for Donut feature_extractor
    processor.feature_extractor.size = {"height":720, "width":960}
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
    model.config.encoder.image_size = (processor.feature_extractor.size["height"], 
                                       processor.feature_extractor.size["width"])
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s>")
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
        # Replace "YourHuggingFaceUsername" with your actual username
        hub_model_id="YourHuggingFaceUsername/donut-base-sroie",
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
