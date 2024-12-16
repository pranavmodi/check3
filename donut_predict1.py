from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import re
from PIL import Image
# Load dataset and get image
# dataset = load_dataset("hf-internal-testing/example-documents", split="test")
# image = dataset[0]["image"]

image = Image.open("test/000.jpg")

# Load model and processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

# Prepare image using processor
pixel_values = processor(image, return_tensors="pt").pixel_values

# Set up the question and prompt
task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
question = "how much is the bill?"
prompt = task_prompt.replace("{user_input}", question)
decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

# Set device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Generate answer
outputs = model.generate(
    pixel_values.to(device),
    decoder_input_ids=decoder_input_ids.to(device),
    max_length=model.decoder.config.max_position_embeddings,
    early_stopping=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    num_beams=1,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
    output_scores=True
)

# Process the output
seq = processor.batch_decode(outputs.sequences)[0]
seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
print("Generated sequence:", seq)

# Convert to JSON format
json_output = processor.token2json(seq)
print("JSON output:", json_output)