from transformers import pipeline

try:
    # Load DeepSeek model for text generation
    predictor = pipeline("text-generation", model="gpt2")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define input text for generation
input_text = "SQL is a"

# Initialize result to None before the try block
result = None  # This ensures result is defined even if the try block fails

try:
    # Generate the next word(s) in the sequence
    result = predictor(
        input_text, 
        max_length=len(input_text.split()) + 5,  # Input length + 5 additional tokens
        num_return_sequences=1
    )
except Exception as e:
    print(f"Error generating text: {e}")
    exit()

# Display generated text
if result:
    print("Generated Text:")
    print(result[0]['generated_text'])
else:
    print("No generated text available.")
