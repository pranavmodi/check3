from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import json
import argparse

def load_model_and_processor():
    """Initialize the model and processor"""
    print("Loading model and processor...")
    
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    
    print("Model and processor loaded successfully")
    return model, processor

def text2json(text):
    """Convert generated text back to JSON format"""
    json_dict = {}
    parts = text.split("<s_")
    
    for part in parts[1:]:  # Skip the first empty part
        key_end = part.find(">")
        if key_end == -1:
            continue
            
        key = part[:key_end]
        value_text = part[key_end + 1:]
        
        end_tag = f"</s_{key}>"
        value_end = value_text.find(end_tag)
        if value_end == -1:
            continue
            
        value = value_text[:value_end].strip()
        json_dict[key] = value
    
    return json_dict

def predict(model, processor, image_path, max_length=512):
    """Perform inference on a single image"""
    print(f"\nProcessing image: {image_path}")
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    
    # Generate text
    generated_ids = model.generate(
        pixel_values,
        max_length=max_length,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.convert_tokens_to_ids("</s>"),
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
    )
    
    # Decode the generated ids to text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print(f"Generated text: {generated_text}")
    
    # Clean and parse the generated text
    generated_text = generated_text.replace("", "").replace("", "")
    parsed_json = text2json(generated_text)
    
    return parsed_json

def main():
    parser = argparse.ArgumentParser(description='Perform inference with Donut model')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to the image file to analyze')
    
    args = parser.parse_args()
    
    # Load model and processor
    model, processor = load_model_and_processor()
    
    # Perform prediction
    result = predict(model, processor, args.image_path)
    
    # Print results
    print("\nExtracted Information:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()