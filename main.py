import os
from flask import Flask, render_template, request, session
import fitz  
import google.generativeai as genai
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Necessary for session management
response_text = ""
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def chatBot():
    
    def isIndexPage(text):
        keywords = ["Introduction","preface"]
        lower_text = text.lower()
        
        # Check for the presence of the keywords
        for keyword in keywords:
            if keyword in lower_text:
                return True
        return False
    
    
    # Main code
    response_text = ""
    final_pgno = 0
    pdf = ""
    if request.method == 'POST':
        files = request.files.getlist("myfile")
        if 'file_paths' not in session:
            session['file_paths'] = []

        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            session['file_paths'].append(file_path)

        greatest = float('-inf')
        final_ans_page = []
        ques = request.form.get('ques')
        question = [ques]
        mode = SentenceTransformer('all-MiniLM-L6-v2')
        final_pgno = 0
        pdf = ""
        checkpoint_count = 0

        for file_path in session['file_paths']:
            filename = os.path.basename(file_path)
            # Opening the PDF file
            doc = fitz.open(file_path)
            pages_list = []
            # Extracting text from each page
            count = 0 
            
            for page in doc:
                count += 1
                text = page.get_text("text")
                text_one_line = text.replace("\n", " ")
                pages_list.append(text_one_line)
                # if checkpoint_count < 2 and isIndexPage(text_one_line):
                #     checkpoint = count
                #     checkpoint_count += 1
                #     continue
            #print(checkpoint)
            # Encoding pages and question
            embeddings = mode.encode(pages_list)
            ques_embeddings = mode.encode(question)
            pgno = 0
            
            for i, embedding in enumerate(embeddings):
                pgno += 1
                embedding = np.array(embedding, dtype=float)
                cosine_sim = cosine_similarity([embedding], ques_embeddings)
                sim_score = cosine_sim[0][0]
                if sim_score > greatest:
                    greatest = sim_score
                    final_ans_page = [pages_list[i]]
                    final_pgno = pgno
                    pdf = filename

        # Printing the text and page number containing the answer
            
        print(f"{final_ans_page} - Page Number {final_pgno} from {pdf}")
        
        # Gemini API Part
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
    return render_template('mult.html', response_text=response_text.replace("*",""), page_num= final_pgno, pdf=pdf)

if __name__ == '__main__':
    app.run(debug=True, port=7000)
