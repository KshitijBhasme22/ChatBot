import os
from flask import Flask, render_template, request
import fitz  
import google.generativeai as genai
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def chatBot():
    response = "Hemlo"
    final_pgno = 5
    if request.method == 'POST':
        files = request.files.getlist("myfile")
        print(files)
        # for file in files:
        #     filename = file.filename
        #     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #     file.save(file_path)
        #     ques = request.form.get('ques')
        #     doc = fitz.open(file_path)
        
        
        
    return render_template('mult.html', response_text=response, page_num= final_pgno)

if __name__ == '__main__':
    app.run(debug=True)