import numpy as np
from tensorflow.keras.models import load_model
import os
from dotenv import load_dotenv

load_dotenv()

# Load your predictive text model
model = load_model("modelv3.h5")

# openai for chatgpt integration
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("API_KEY"),
)
 
import json 
unique_token_index = None
# load the tokens from the JSON file
with open("./unique_token_index.json", 'r') as json_file:
    unique_token_index = json.load(json_file)

n_words = 10
unique_tokens = list(unique_token_index)


# Function to predict the next word using your model
def predict_next_word(input_text, n_best):
    input_text = input_text.lower()
    X = np.zeros((1, n_words, len(unique_tokens)))
    for i, word in enumerate(input_text.split()):
        if word in unique_token_index:
            X[0, i, unique_token_index[word]] = True
    predictions = model.predict(X)[0]
    return [unique_tokens[idx] for idx in np.argpartition(predictions, -n_best)[-n_best:]]



# Function to generate the rest of the sentence using ChatGPT
def complete_sentence(prompt):
    # Specify the chat conversation
    message = [
        {"role": "user", "content": prompt}
    ]
    # Create a chat completion using the OpenAI GPT-3.5-turbo model
    response = client.completions.create(
        prompt=message,
        model="gpt-3.5-turbo",
        top_p=0.5, 
        max_tokens=50,
        stream=True)
    return response.choices[0].text or ""

# Main loop for interaction
while True:
    # enters a partial sentence
    user_input = input("Enter a partial sentence [q for quit]: ")

    if user_input.strip() == "q":
        break
    # Predict the next word using your model
    next_word_predictions = predict_next_word(user_input, 3)
    # Form a new prompt with the predicted next words
    prompt_with_predictions = f"{user_input} {next_word_predictions[0]}"
    print("Next word Predicted sentence:$>", prompt_with_predictions)
    # Generate the rest of the sentence using ChatGPT
    completion = complete_sentence(prompt_with_predictions)
    # Display the completed sentence
    print(f"Completed Sentence:$> {user_input} {next_word_predictions[0]} {completion}")
