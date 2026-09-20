import os
import torch
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

app = Flask(__name__)
# Enable Cross-Origin Resource Sharing so our free static page can securely talk to it
CORS(app)

class TabulaRasaKnowledgeCore:
    def __init__(self):
        print("📥 Initializing free local Embedding Engine (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.pdf_vault = []          
        self.pdf_embeddings = None    
        self.chat_memory_vault = {}  
        self.total_tokens_processed = 0

    def add_conversational_memory(self, user_phrase: str, target_reply: str):
        clean_input = user_phrase.strip().lower()
        clean_target = target_reply.strip()
        self.chat_memory_vault[clean_input] = clean_target
        self.total_tokens_processed += len(clean_input.split())
        return f"🧠 Memory Linked! Saved '{clean_input}' to retrieve '{clean_target}'."

    def digest_pdf(self, file_path: str):
        try:
            reader = PdfReader(file_path)
            text_chunks = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    chunks = [c.strip() for c in text.split('.') if len(c.strip()) > 10]
                    text_chunks.extend(chunks)
                    
            if not text_chunks:
                return False
                
            self.pdf_vault.extend(text_chunks)
            self.pdf_embeddings = self.embedding_model.encode(self.pdf_vault, convert_to_tensor=True)
            self.total_tokens_processed += len(text_chunks)
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def query_pdf_vault(self, query: str) -> str:
        if self.pdf_embeddings is None or not self.pdf_vault:
            return "No documents uploaded in memory yet."
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        cos_scores = torch.nn.functional.cosine_similarity(query_embedding, self.pdf_embeddings)
        best_match_idx = torch.argmax(cos_scores).item()
        if cos_scores[best_match_idx] > 0.25: 
            return self.pdf_vault[best_match_idx]
        return "I couldn't find a strong mathematical match inside the uploaded document."

    def search_live_internet(self, query: str) -> str:
        try:
            url = f"https://duckduckgo.com{query}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                snippets = [span.text for span in soup.find_all('span', class_='zci__result__snippet') if span.text]
                if snippets: return "\n\n".join(snippets[:3])
                results = [a.text for a in soup.find_all('a', class_='result__snippet') if a.text]
                if results: return "\n\n".join(results[:3])
            return "Web server reached, but no clean snippet summary could be parsed."
        except Exception as e:
            return f"Network routing issue: {str(e)}"

ai_child = TabulaRasaKnowledgeCore()

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"status": "No file chunk mapped."})
    file = request.files['file']
    save_path = "temp.pdf"
    file.save(save_path)
    success = ai_child.digest_pdf(save_path)
    if os.path.exists(save_path): os.remove(save_path)
    msg = "🏆 PDF successfully parsed and indexed!" if success else "❌ Error parsing document."
    return jsonify({"status": msg, "tokens": ai_child.total_tokens_processed})

@app.route('/interact', methods=['POST'])
def interact():
    data = request.json
    mode = data.get('mode')
    payload = data.get('payload', '').strip()
    
    if mode == "LEARN":
        if " -> " in payload:
            input_side, target_side = payload.split(" -> ", 1)
            reply_msg = ai_child.add_conversational_memory(input_side, target_side)
        else: reply_msg = "⚠️ Formatting error! Use: input -> output"
    elif mode == "RAG": reply_msg = ai_child.query_pdf_vault(payload)
    elif mode == "WEB": reply_msg = ai_child.search_live_internet(payload)
    else:
        lookup = payload.lower()
        reply_msg = f"🤖 Output: '{ai_child.chat_memory_vault[lookup]}'" if lookup in ai_child.chat_memory_vault else "🤖 Unmapped word sequence."

    return jsonify({"reply": reply_msg, "tokens": ai_child.total_tokens_processed})

if __name__ == '__main__':
    # Render maps internal server bindings using individual environment port hooks
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
