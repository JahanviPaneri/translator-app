# Install required libraries (run this only once)
# pip install torch transformers sentencepiece

from transformers import MarianTokenizer, MarianMTModel

# Load a pre-trained English -> Hindi translation model
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Function to translate text
def translate(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    # Generate translation
    translated_tokens = model.generate(**inputs)
    # Decode back to text
    translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translation

# Simple translator app
while True:
    user_input = input("Enter text in English (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    output = translate(user_input)
    print("Hindi Translation:", output)
