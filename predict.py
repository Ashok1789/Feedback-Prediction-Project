from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import google.generativeai as genai
from config import GOOGLE_API_KEY
from google.cloud import speech
def voice_diarization(audio_file_path):
    audio_file = r"C:\Users\ashok\Music\predict\negative 2.mp3"
    client = speech.SpeechClient.from_service_account_file(r"C:\Users\ashok\Music\predict\json\amazing-hub-416612-91366d853f2e.json")
    with open(audio_file, 'rb') as f:
        mp3_data = f.read()
    audio_file = speech.RecognitionAudio(content=mp3_data)
    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization = True,
    )
    config = speech.RecognitionConfig(
        sample_rate_hertz = 44100,
        language_code = 'en-US',
        diarization_config = diarization_config,
        enable_automatic_punctuation = True,
    )
    response = client.recognize(config = config, audio = audio_file)
    result = response.results[-1]
    words_info = result.alternatives[0].words
    speaker1 = ''
    speaker2 = ''
    for word_info in words_info:
        if word_info.speaker_tag == 1:
            speaker1 = speaker1 + ' ' + word_info.word
        else :
            speaker2 = speaker2 + ' ' + word_info.word

    customer_salesman(speaker1, speaker2)
    rating, response = customer_salesman(speaker1, speaker2)
    diarization_result = "Diarization completed successfully. Result: ..."
    return diarization_result,rating,response
def customer_salesman(speaker1, speaker2):
  from config import GOOGLE_API_KEY
  genai.configure(api_key=GOOGLE_API_KEY)
  google_model = genai.GenerativeModel('gemini-pro')
  prompt = "Speaker 1: {0}, Speaker 2: {1} .Find out the Customer between these two speakers. I want you to reply like Speaker 1 or Speaker 2 nothing else. Just one word.".format(speaker1, speaker2)
  response = google_model.generate_content(prompt)
  speaker = response.text
  if speaker == "Speaker 1":
    classifier(speaker1)

  else:
    classifier(speaker2)
    rating, response = customer_salesman(speaker1, speaker2)
    diarization_result = "Diarization completed successfully. Result: ..."
    return diarization_result,rating,response
def classifier(text):
  MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
  tokenizer = AutoTokenizer.from_pretrained(MODEL)
  model = AutoModelForSequenceClassification.from_pretrained(MODEL)
  encoded_text = tokenizer(text, return_tensors='pt')
  output = model(**encoded_text)
  scores = output[0][0].detach().numpy()
  scores = softmax(scores)
  scores_dict = {
      'roberta_neg' : scores[0],
      'roberta_neu' : scores[1],
      'roberta_pos' : scores[2]
  }
  if scores_dict['roberta_neg'] > scores_dict['roberta_pos']:
    reason_extractor(text, 'Negative')
  else:
    reason_extractor(text, 'Positive')

def reason_extractor(text,rating):
  from config import GOOGLE_API_KEY
  genai.configure(api_key=GOOGLE_API_KEY)
  google_model = genai.GenerativeModel('gemini-pro')
  prompt = "Text : {0}. I want you to extract the reasons why this product is rated {1} based on the text only. I want the output to be in the format : The customer loves or hates the product because [reasons]. I want the answer to be short in one or two lines with the essentials reasons only".format(text, rating)
  response = google_model.generate_content(prompt)

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from functools import partial
from predict import voice_diarization

def execute_diarization(audio_file_path):
    try:
        # Execute voice diarization function and get the result, rating, and response
        result, rating, response = voice_diarization(audio_file_path)
        # Update the text widget with the diarization result, rating, and response
        text_output.delete(1.0, tk.END)  # Clear previous content
        text_output.insert(tk.END, f"Diarization Result: {result}\n")
        text_output.insert(tk.END, f"Rating: {rating}\n")
        text_output.insert(tk.END, f"Response: {response}\n")
        messagebox.showinfo("Success", "Voice diarization completed successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3")])
    if file_path:
        entry_file_path.delete(0, tk.END)
        entry_file_path.insert(0, file_path)

def start_diarization():
    audio_file_path = entry_file_path.get()
    if not audio_file_path:
        messagebox.showwarning("Warning", "Please select an audio file!")
        return
    execute_diarization(audio_file_path)
    # Show message when diarization completes
    messagebox.showinfo("Diarization", "Diarization process initiated. Please wait for completion.")

# Create main window
root = tk.Tk()
root.title("Review App")

# Create file selection widgets
label_file_path = tk.Label(root, text="Audio File Path:")
label_file_path.grid(row=0, column=0, padx=5, pady=5)

entry_file_path = tk.Entry(root, width=50)
entry_file_path.grid(row=0, column=1, padx=5, pady=5)

button_browse = tk.Button(root, text="Browse", command=browse_file)
button_browse.grid(row=0, column=2, padx=5, pady=5)

# Create start button
button_start = tk.Button(root, text="Start Diarization", command=start_diarization)
button_start.grid(row=1, column=1, padx=5, pady=5)

# Create text widget to display output
text_output = tk.Text(root, height=10, width=60)
text_output.grid(row=2, columnspan=3, padx=5, pady=5)

root.mainloop()