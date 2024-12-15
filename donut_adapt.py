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
import argparse
import datasets

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

    # Debug prints
    print(f"Looking for metadata files in: {metadata_path}")
    print(f"Looking for images in: {image_path}")
    
    # Check if directories exist
    if not metadata_path.exists():
        raise ValueError(f"Metadata directory not found: {metadata_path}")
    if not image_path.exists():
        raise ValueError(f"Image directory not found: {image_path}")

    # Count files
    metadata_files = list(metadata_path.glob("*.json"))
    image_files = list(image_path.glob("*.jpg"))
    print(f"Found {len(metadata_files)} JSON files and {len(image_files)} image files")

    # Parse metadata
    for file_name in metadata_files:
        try:
            with open(file_name, "r") as json_file:
                data = json.load(json_file)
                text = json.dumps(data)
                image_file = image_path.joinpath(f"{file_name.stem}.jpg")
                if image_file.is_file():
                    metadata_list.append({
                        "text": text,
                        "file_name": f"{file_name.stem}.jpg"
                    })
                else:
                    print(f"Warning: No matching image file for {file_name.stem}")
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON file {file_name}")
            continue
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            continue

    if not metadata_list:
        raise ValueError("No valid metadata entries found! Check the paths and file contents above.")

    print(f"Successfully processed {len(metadata_list)} metadata entries")

    # Create metadata.jsonl in the image directory
    metadata_file = image_path.joinpath("metadata.jsonl")
    with open(metadata_file, 'w') as outfile:
        for entry in metadata_list:
            json.dump(entry, outfile)
            outfile.write('\n')

    print(f"Created metadata file at: {metadata_file}")

    # Load the dataset from imagefolder with correct feature specification
    from datasets import Features, Value, Image
    
    features = Features({
        'image': Image(),
        'text': Value('string'),
        'file_name': Value('string')
    })

    dataset = load_dataset(
        "imagefolder", 
        data_dir=image_path, 
        split="train",
        features=features
    )

    # Add text information
    text_map = {item["file_name"]: item["text"] for item in metadata_list}
    
    def add_text(example):
        fname = example["file_name"]
        example["text"] = text_map.get(fname, json.dumps({}))
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

def setup_environment():
    """Set up environment variables and HF token."""
    load_dotenv()
    token = os.getenv('HF_TOKEN')
    if not token:
        raise ValueError("Please ensure HF_TOKEN is set in your .env file")
    HfFolder.save_token(token)
    print("✓ Successfully set up Hugging Face token")

def initialize_processor(dataset):
    """Initialize and configure the Donut processor with special tokens."""
    # Gather all possible keys from the dataset's JSON texts
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

    # Initialize and configure processor
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    processor.tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
    processor.feature_extractor.size = {"height":720, "width":960}
    processor.feature_extractor.do_align_long_axis = False
    
    return processor

def process_dataset(dataset, processor, batch_size):
    """Process and transform the dataset."""
    proc_dataset = dataset.map(
        preprocess_documents,
        batched=True,
        batch_size=batch_size
    )

    processed_dataset = proc_dataset.map(
        lambda x: transform_and_tokenize(x, processor),
        remove_columns=["image", "text"],
        batched=True,
        batch_size=batch_size
    )

    return processed_dataset.train_test_split(test_size=0.1)

def setup_model(processor, checkpoint_path=None):
    """Initialize and configure the model."""
    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path)
    else:
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    # Configure model settings
    model.config.encoder.image_size = (processor.feature_extractor.size["height"], 
                                     processor.feature_extractor.size["width"])
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s>")
    
    return model

def get_training_args(args):
    """Set up and return training arguments."""
    # Check if CUDA is available
    import torch
    use_fp16 = torch.cuda.is_available()  # Only use fp16 if running on GPU
    
    return Seq2SeqTrainingArguments(
        output_dir=args.checkpoint_dir,
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        weight_decay=0.01,
        fp16=use_fp16,  # Now conditional on GPU availability
        logging_steps=50,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        predict_with_generate=True,
        report_to="tensorboard",
        hub_strategy="every_save" if args.push_to_hub else "end",
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        push_to_hub=args.push_to_hub,
        hub_token=HfFolder.get_token() if args.push_to_hub else None,
    )

def train_model(model, training_args, train_dataset, test_dataset):
    """Initialize trainer and train the model."""
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    trainer.train()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Donut model with custom batch size')
    parser.add_argument('--batch_size', 
                       type=int, 
                       default=2,
                       help='Batch size for training (default: 2)')
    parser.add_argument('--load_saved',
                       action='store_true',
                       help='Load previously processed data instead of processing again')
    parser.add_argument('--checkpoint_dir',
                       type=str,
                       default="donut-base-sroie",
                       help='Directory to save checkpoints')
    parser.add_argument('--push_to_hub',
                       action='store_true',
                       help='Push checkpoints to Hugging Face Hub')
    parser.add_argument('--hub_model_id',
                       type=str,
                       help='Hugging Face Hub model ID (username/model-name)')
    parser.add_argument('--checkpoint_path',
                       type=str,
                       help='Path to load model checkpoint from')
    return parser.parse_args()

def save_processed_data(dataset, processor, output_dir="processed_data"):
    """Save the processed dataset and processor to disk."""
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the dataset
    print(f"\n💾 Saving processed dataset to {output_path}...")
    dataset.save_to_disk(output_path / "dataset")

    # Save the processor
    print(f"💾 Saving processor to {output_path}...")
    processor.save_pretrained(output_path / "processor")
    print("✓ Successfully saved processed data and processor")

def load_processed_data(input_dir="processed_data"):
    """Load the processed dataset and processor from disk."""
    input_path = Path(input_dir)
    
    # Load the dataset
    print(f"\n📂 Loading processed dataset from {input_path}...")
    dataset = datasets.load_from_disk(input_path / "dataset")
    
    # Load the processor
    print(f"📂 Loading processor from {input_path}...")
    processor = DonutProcessor.from_pretrained(input_path / "processor")
    print("✓ Successfully loaded processed data and processor")
    
    return dataset, processor

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set up environment
    print("\n🔧 Setting up environment...")
    setup_environment()

    if args.load_saved:
        print("\n📂 Loading previously processed data...")
        train_test_dataset, processor = load_processed_data()
        print(f"✓ Loaded dataset with {len(train_test_dataset['train'])} train and {len(train_test_dataset['test'])} test samples")
    else:
        # Prepare dataset
        print("\n📁 Preparing dataset...")
        dataset = prepare_dataset()
        print(f"✓ Dataset prepared with {len(dataset)} samples")

        # Initialize processor
        print("\n🔄 Initializing Donut processor...")
        processor = initialize_processor(dataset)
        print("✓ Processor initialized and configured")

        # Process dataset
        print("\n🔄 Processing and transforming dataset...")
        train_test_dataset = process_dataset(dataset, processor, args.batch_size)
        print(f"✓ Dataset split into {len(train_test_dataset['train'])} train and {len(train_test_dataset['test'])} test samples")

        # Save processed data
        save_processed_data(train_test_dataset, processor)

    # Setup model
    print("\n🔄 Setting up model...")
    model = setup_model(processor, args.checkpoint_path)
    print("✓ Model initialized and configured")

    # Get training arguments
    print(f"\n⚙️ Setting up training arguments with batch size {args.batch_size}...")
    training_args = get_training_args(args)
    print("✓ Training arguments configured")

    # Train model
    print("\n🚀 Starting training...")
    train_model(model, training_args, train_test_dataset["train"], train_test_dataset["test"])
    print("✓ Training completed!")

if __name__ == "__main__":
    main()
