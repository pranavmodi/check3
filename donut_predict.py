from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import json
import argparse

class DonutPredictor:
    def __init__(self, model_path, processor_path):
        """
        Initialize the Donut predictor with trained model and processor
        
        Args:
            model_path: Path to the trained model
            processor_path: Path to the saved processor
        """
        self.processor = DonutProcessor.from_pretrained(processor_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        
    def predict(self, image_path, max_length=512):
        """
        Perform inference on a single image
        
        Args:
            image_path: Path to the image file
            max_length: Maximum length of the generated sequence
            
        Returns:
            Parsed JSON output
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Generate text
        generated_ids = self.model.generate(
            pixel_values,
            max_length=max_length,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.convert_tokens_to_ids(""),
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
        )
        
        # Decode the generated ids to text
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Parse the generated text to JSON
        # Remove the task start token and end token
        generated_text = generated_text.replace("", "").replace("", "")
        
        # Convert the generated text back to JSON structure
        parsed_json = self.text2json(generated_text)
        
        return parsed_json
    
    def text2json(self, text):
        """
        Convert generated text back to JSON format
        """
        json_dict = {}
        current_key = None
        parts = text.split("<s_")
        
        for part in parts[1:]:  # Skip the first empty part
            # Split at the first occurrence of ">"
            key_end = part.find(">")
            if key_end == -1:
                continue
                
            key = part[:key_end]
            value_text = part[key_end + 1:]
            
            # Find the end tag for this key
            end_tag = f"</s_{key}>"
            value_end = value_text.find(end_tag)
            if value_end == -1:
                continue
                
            value = value_text[:value_end].strip()
            json_dict[key] = value
            
        return json_dict

def main():
    parser = argparse.ArgumentParser(description='Perform inference with Donut model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--processor_path', type=str, required=True,
                        help='Path to the saved processor')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the image file to analyze')
    
    args = parser.parse_args()
    
    predictor = DonutPredictor(
        model_path=args.model_path,
        processor_path=args.processor_path
    )
    
    result = predictor.predict(args.image_path)
    print("Extracted Information:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()