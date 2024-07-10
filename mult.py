import os
from flask import Flask, render_template, request
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai

app = Flask(__name__)
response_text = ""
UPLOAD_FOLDER = 'uploads/'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def chatBot():
    # Section to identify the index page
    def index_page(text):
        keywords = ["table of contents", "contents", "index"]
        lower_text = text.lower()
        for keyword in keywords:
            if keyword in lower_text:
                return True
        return False

    final_ans_page = ""
    final_pgno = 0
    response_text = ""
    checkpoint = 0  # used to find the page numbers till index page

    if request.method == 'POST':
        model = SentenceTransformer('all-MiniLM-L6-v2')
        files = request.files.getlist("myfile")
        ques = request.form.get('ques')
        question = [ques]
        ques_embeddings = model.encode(question)
        greatest = float('-inf')

        for file in files:
            if file.filename == '':
                response_text = "No file selected for uploading!"
                return render_template('mult.html', response_text=response_text)

            if file:
                filename = file.filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Opening the PDF file
                doc = fitz.open(file_path)

                # Extracting text from each page
                count = 0
                pgno = 0
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text("text")
                    text_one_line = text.replace("\n", " ")
                    embeddings = model.encode([text_one_line])
                    cosine_sim = cosine_similarity(embeddings, ques_embeddings)
                    sim_score = cosine_sim[0][0]
                    if sim_score > greatest:
                        greatest = sim_score
                        final_ans_page = text_one_line
                        final_pgno = pgno  # setting the page number for the answer text page
                    if checkpoint == 0 and index_page(text_one_line):
                        checkpoint = count

        final_pgno = final_pgno - checkpoint
        print(f"{final_ans_page} - Page Number {final_pgno}")

        # Assuming you have set up the API key and necessary configurations for Generative AI
        api_key = "AIzaSyB39vi0wJ7Rwt1IkLARVbTDDOigHXKaEsM"
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )

        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        f"Remember this data {final_ans_page}"
                    ],
                },
            ]
        )
        try:
            response = chat_session.send_message(f"{ques}")
            response_text = response.text
        except Exception as e:
            print(f"{e} - Please try something else!")
            response_text = "An error occurred!"

    return render_template('mult.html', response_text=response_text, page_num=final_pgno)

if __name__ == '__main__':
    app.run(debug=True, port=9000)
