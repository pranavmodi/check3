from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import json
import argparse

class DonutPredictor:
    def __init__(self, model_path, processor_path=None, use_hf=False):
        """
        Initialize the Donut predictor with trained model and processor
        """
        print(f"Initializing DonutPredictor with model_path: {model_path}")
        
        # Check if processor path is provided
        if processor_path:
            print(f"Using provided processor path: {processor_path}")
            if use_hf:
                print("Loading processor from Hugging Face...")
                self.processor = DonutProcessor.from_pretrained(processor_path, trust_remote_code=True)
            else:
                print("Loading processor from local path...")
                self.processor = DonutProcessor.from_pretrained(processor_path)
        else:
            print("No processor path provided, using default donut-base processor...")
            self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        
        # Load the model
        if use_hf:
            print("Loading model from Hugging Face...")
            self.model = VisionEncoderDecoderModel.from_pretrained(model_path, trust_remote_code=True)
        else:
            print("Loading model from local path...")
            self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        
        print("Model and processor loaded successfully")
        
        print("All special tokens:")
        print(self.processor.tokenizer.all_special_tokens)
        print("\nSpecial tokens mapping:")
        print(self.processor.tokenizer.special_tokens_map)
        
    def predict(self, image_path, max_length=512):
        """
        Perform inference on a single image
        """
        print(f"\nProcessing image: {image_path}")
        
        # Load and preprocess image
        print("Loading and converting image to RGB...")
        image = Image.open(image_path).convert("RGB")
        print("Preprocessing image with processor...")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        print(f"Image preprocessed, tensor shape: {pixel_values.shape}")
        
        # Generate text
        print("\nGenerating text from image...")
        generated_ids = self.model.generate(
            pixel_values,
            max_length=max_length,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            # eos_token_id=self.processor.tokenizer.convert_tokens_to_ids(""),
            eos_token_id = self.processor.tokenizer.convert_tokens_to_ids("</s>"),
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
        )
        print(f"Generated ids shape: {generated_ids.shape}")
        
        # Decode the generated ids to text
        print("\nDecoding generated ids to text...")
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        print(f"Raw generated text: {generated_text}")
        
        # Parse the generated text to JSON
        print("\nCleaning generated text...")
        generated_text = generated_text.replace("", "").replace("", "")
        print(f"Cleaned text: {generated_text}")
        
        # Convert the generated text back to JSON structure
        print("\nConverting text to JSON...")
        parsed_json = self.text2json(generated_text)
        print(f"Parsed JSON: {json.dumps(parsed_json, indent=2)}")
        
        return parsed_json
    
    def text2json(self, text):
        """
        Convert generated text back to JSON format
        """
        print("\nStarting text to JSON conversion...")
        json_dict = {}
        current_key = None
        parts = text.split("<s_")
        print(f"Found {len(parts)-1} parts to process")
        
        for i, part in enumerate(parts[1:], 1):  # Skip the first empty part
            print(f"\nProcessing part {i}/{len(parts)-1}")
            # Split at the first occurrence of ">"
            key_end = part.find(">")
            if key_end == -1:
                print(f"No closing '>' found in part {i}, skipping...")
                continue
                
            key = part[:key_end]
            value_text = part[key_end + 1:]
            print(f"Found key: {key}")
            
            # Find the end tag for this key
            end_tag = f"</s_{key}>"
            value_end = value_text.find(end_tag)
            if value_end == -1:
                print(f"No closing tag found for key '{key}', skipping...")
                continue
                
            value = value_text[:value_end].strip()
            json_dict[key] = value
            print(f"Added key-value pair: {key}: {value}")
            
        return json_dict

    @staticmethod
    def initialize_processor(dataset_json_texts):
        """
        Initialize and configure the Donut processor with special tokens from dataset.
        
        Args:
            dataset_json_texts (list): List of JSON strings from the dataset
        
        Returns:
            DonutProcessor: Configured processor with special tokens
        """
        print("Initializing new Donut processor...")
        
        # Gather all possible keys from the dataset's JSON texts
        all_keys = set()
        for text_str in dataset_json_texts:
            try:
                data = json.loads(text_str)
                all_keys.update(DonutPredictor.gather_json_keys(data))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON string: {text_str}")
        
        # Build special tokens for keys
        new_special_tokens = ["<s>", "</s>"]  # basic tokens
        for k in all_keys:
            new_special_tokens.append(f"<s_{k}>")
            new_special_tokens.append(f"</s_{k}>")
        new_special_tokens.append("<sep/>")
        
        print(f"Found {len(all_keys)} unique keys")
        print(f"Created {len(new_special_tokens)} special tokens")
        
        # Initialize and configure processor
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        processor.tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
        processor.feature_extractor.size = {"height": 720, "width": 960}
        processor.feature_extractor.do_align_long_axis = False
        
        print("Processor initialized successfully")
        return processor

    @staticmethod
    def gather_json_keys(json_obj, prefix=""):
        """
        Recursively gather all keys from a JSON object.
        
        Args:
            json_obj: JSON object to process
            prefix: Prefix for nested keys
            
        Returns:
            set: Set of all keys found in the JSON object
        """
        keys = set()
        if isinstance(json_obj, dict):
            for key, value in json_obj.items():
                full_key = f"{prefix}{key}" if prefix else key
                keys.add(full_key)
                if isinstance(value, (dict, list)):
                    keys.update(DonutPredictor.gather_json_keys(value, f"{full_key}."))
        elif isinstance(json_obj, list):
            for item in json_obj:
                keys.update(DonutPredictor.gather_json_keys(item, prefix))
        return keys

def main():
    parser = argparse.ArgumentParser(description='Perform inference with Donut model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model or HF model name')
    parser.add_argument('--processor_path', type=str, required=False, default=None,
                        help='Path to the saved processor or HF processor name')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the image file to analyze')
    parser.add_argument('--use_hf', action='store_true',
                        help='Load model and processor from Hugging Face')
    
    args = parser.parse_args()
    print("\nStarting Donut prediction with arguments:")
    print(f"Model path: {args.model_path}")
    print(f"Processor path: {args.processor_path}")
    print(f"Image path: {args.image_path}")
    print(f"Use HF: {args.use_hf}")
    
    predictor = DonutPredictor(
        model_path=args.model_path,
        processor_path=args.processor_path,
        use_hf=args.use_hf
    )
    
    result = predictor.predict(args.image_path)
    print("\nFinal Extracted Information:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()