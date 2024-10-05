from otter.modeling_otter import OtterForConditionalGeneration

from transformers import AutoModelForCausalLM, AutoTokenizer

# Model name
model_name = "Charliebear/BrainGPT"

# # Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
#
# # Check if the model is successfully loaded
# print(f"Loaded {model_name} successfully!")


model = OtterForConditionalGeneration.from_pretrained(model_name)
model.text_tokenizer.padding_side = "left"
tokenizer = model.text_tokenizer
image_processor = transformers.CLIPImageProcessor()
model.eval()
