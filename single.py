import os
from flask import Flask, render_template, request
import fitz  
import google.generativeai as genai
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
app = Flask(__name__)
response_text = ""
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
global pgno
@app.route('/', methods=['GET', 'POST'])
def chatBot():
    # Section is identifying the index page
    def index_page(text):
        keywords = ["table of contents", "contents", "index"]
        lower_text = text.lower()
        
        # Check for the presence of any keyword
        for keyword in keywords:
            if keyword in lower_text:
                return True
        return False
    
    # Main code
    final_pgno = 0
    
    response_text = ""
    if request.method == 'POST':
        file = request.files['myfile']
        
        if file.filename == '':
            response_text = "No file selected for uploading!"
        
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            ques = request.form.get('ques')
            api_key = "AIzaSyDDqgf8N8yB_WfaTDzOkfiGWoaGjPUzPPQ"  
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

            # Opening the PDF file
            doc = fitz.open(file_path)
            pages_list = []

            # Extracting text from each page
            checkpoint = 0 # used to find the page numbers till index page
            count = 0 
            for page in doc:
                count+=1
                text = page.get_text("text")
                text_one_line = text.replace("\n", " ")
                pages_list.append(text_one_line)
                if checkpoint==0 and index_page(text_one_line)==True:
                    checkpoint = count
                    continue
                
            question = [ques]

            # Encoding pages and question
            mode = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = mode.encode(pages_list)
            ques_embeddings = mode.encode(question)
            
            
            pgno = 0
            final_ans_page = []
            greatest = float('-inf')
            for i, embedding in enumerate(embeddings):
                pgno+=1
                embedding = np.array(embedding, dtype=float)
                cosine_sim = cosine_similarity([embedding], ques_embeddings)
                sim_score = cosine_sim[0][0]
                if sim_score > greatest:
                    greatest = sim_score
                    final_ans_page = [pages_list[i]]
                    final_pgno = pgno #setting the page number for the answer text page
            final_pgno = final_pgno - checkpoint
            # Printing the text and page number containing the answer
            print(f"{final_ans_page} - Page Number {final_pgno}")
            
                
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
            except Exception as e:
                print(f"{e} - Please try something else!")
            response_text = response.text
    return render_template('index.html', response_text=response_text, page_num= final_pgno)

if __name__ == '__main__':
    app.run(debug=True)