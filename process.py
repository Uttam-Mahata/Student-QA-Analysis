import os
import spacy
from spacy.pipeline import Sentencizer

# Load the English language model in spaCy
nlp = spacy.load("en_core_web_sm")

# Create a Sentencizer component
sentencizer = Sentencizer()

# Add the Sentencizer component to the spaCy pipeline
nlp.add_pipe('sentencizer')

# Define custom exceptions for spaCy tokenization
custom_exceptions = ["P.", "V.", "Dr.", "Mr.", "Mrs.", "Ms.", "etc."]  # Add more as needed

def process_answers(folder_path):
    # Iterate through each question folder
    for question_folder in os.listdir(folder_path):
        question_folder_path = os.path.join(folder_path, question_folder)
        # Check if the item in the folder is a directory
        if os.path.isdir(question_folder_path):
            # Iterate through each answer file in the question folder
            for answer_file in os.listdir(question_folder_path):
                answer_file_path = os.path.join(question_folder_path, answer_file)
                # Check if the item in the folder is a file
                if os.path.isfile(answer_file_path):
                    # Open the answer file for reading
                    with open(answer_file_path, 'r') as f:
                        # Read the content of the file
                        content = f.read()
                    # Process the content using spaCy
                    doc = nlp(content)
                    # Extract sentences from the parsed document
                    sentences = []
                    current_sentence = ""
                    for token in doc:
                        if token.text in custom_exceptions:
                            current_sentence += token.text_with_ws
                        elif token.is_sent_start:
                            if current_sentence:
                                sentences.append(current_sentence.strip())
                                current_sentence = ""
                            current_sentence += token.text_with_ws
                        else:
                            current_sentence += token.text_with_ws
                    if current_sentence:
                        sentences.append(current_sentence.strip())
                    # Write the processed sentences back to the file
                    with open(answer_file_path, 'w') as f:
                        for sentence in sentences:
                            f.write(sentence + '\n')

# Specify the path to the answers folder
answers_folder_path = 'answers'

# Call the function to process the answers
process_answers(answers_folder_path)
