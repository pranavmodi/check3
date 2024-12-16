from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import json
import argparse

class DonutPredictor:
    def __init__(self, model_path, processor_path, use_hf=False):
        """
        Initialize the Donut predictor with trained model and processor
        """
        print(f"Initializing DonutPredictor with model_path: {model_path}, processor_path: {processor_path}")
        if use_hf:
            print("Loading model and processor from Hugging Face...")
            self.processor = DonutProcessor.from_pretrained(processor_path, trust_remote_code=True)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_path, trust_remote_code=True)
        else:
            print("Loading model and processor from local path...")
            self.processor = DonutProcessor.from_pretrained(processor_path)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        print("Model and processor loaded successfully")
        
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
            eos_token_id=self.processor.tokenizer.convert_tokens_to_ids(""),
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

def main():
    parser = argparse.ArgumentParser(description='Perform inference with Donut model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model or HF model name')
    parser.add_argument('--processor_path', type=str, required=True,
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