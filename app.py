import torch
import json
import requests
import os
from flask import Flask, render_template_string, request, jsonify
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# -------------------------------------------------------------
# 🧠 THE UPGRADED CORE ENVIRONMENT WITH MEMORY & RAG
# -------------------------------------------------------------
class TabulaRasaKnowledgeCore:
    def __init__(self):
        print("📥 Initializing free local Embedding Engine (all-MiniLM-L6-v2)...")
        # Downloads a completely free local encoder model to convert text to math vectors
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Internal memory storage systems
        self.pdf_vault = []           # Holds text chunks extracted from uploaded documents
        self.pdf_embeddings = None    # Matrix of document math representations
        self.chat_memory_vault = {}   # Direct word-pair mappings from conversational training
        
        # Configuration settings
        self.total_tokens_processed = 0
        self.loss_history = [0.0]

    def add_conversational_memory(self, user_phrase: str, target_reply: str):
        """LEARNING MODE: Maps simple phrase parameters to targets directly."""
        clean_input = user_phrase.strip().lower()
        clean_target = target_reply.strip()
        self.chat_memory_vault[clean_input] = clean_target
        self.total_tokens_processed += len(clean_input.split())
        self.loss_history.append(0.01) # Simulated convergence log
        return 0.01

    def digest_pdf(self, file_path: str):
        """RAG COMPONENT: Extracts text blocks and builds a local math index grid."""
        reader = PdfReader(file_path)
        text_chunks = []
        
        # Read the document page by page and split it into clean sentences
        for page in reader.pages:
            text = page.extract_text()
            if text:
                chunks = [c.strip() for c in text.split('.') if len(c.strip()) > 10]
                text_chunks.extend(chunks)
                
        if not text_chunks:
            return False
            
        self.pdf_vault.extend(text_chunks)
        # Convert text chunks to structural tensor arrays
        embeddings_array = self.embedding_model.encode(self.pdf_vault, convert_to_tensor=True)
        self.pdf_embeddings = embeddings_array
        self.total_tokens_processed += len(text_chunks) * 5
        return True

    def query_pdf_vault(self, query: str) -> str:
        """RAG COMPONENT: Performs cosine matrix similarity calculations locally."""
        if self.pdf_embeddings is None or not self.pdf_vault:
            return "No documents uploaded in memory yet."
            
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        # Standard PyTorch matrix dot-product to compute alignment scores
        cos_scores = torch.nn.functional.cosine_similarity(query_embedding, self.pdf_embeddings)
        best_match_idx = torch.argmax(cos_scores).item()
        
        if cos_scores[best_match_idx] > 0.25: # Strict alignment threshold validation
            return self.pdf_vault[best_match_idx]
        return "I couldn't find a strong mathematical match inside the uploaded document."

    def search_live_internet(self, query: str) -> str:
        """INTERNET CONNECTIVITY: Pulls search data using a free proxy search service."""
        try:
            # Employs a public DuckDuckGo text tracking URL structure to avoid paid Google API keys
            url = f"https://duckduckgo.com{query}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                snippets = [span.text for span in soup.find_all('span', class_='zci__result__snippet') if span.text]
                if snippets:
                    return snippets[0]
                # Secondary fallback extractor pattern
                results = [a.text for a in soup.find_all('a', class_='result__snippet') if a.text]
                if results:
                    return results[0]
            return "Web server reached, but no clean snippet summary could be parsed."
        except Exception as e:
            return f"Network routing issue: {str(e)}"

# Instantiate the Upgraded Global Framework System
ai_child = TabulaRasaKnowledgeCore()

# -------------------------------------------------------------
# 🌐 THE FRONTEND CONSOLE ENGINE
# -------------------------------------------------------------
HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tabula Rasa Advanced RAG System</title>
    <style>
        :root { --bg: #0f172a; --panel: #1e293b; --accent: #38bdf8; --text: #f8fafc; --text-mut: #94a3b8; }
        body { font-family: sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; display: flex; justify-content: center; }
        .container { width: 100%; max-width: 1000px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .panel { background: var(--panel); border-radius: 12px; padding: 20px; border: 1px solid #334155; display: flex; flex-direction: column; }
        h2 { margin-top: 0; font-size: 1.2rem; border-bottom: 2px solid #334155; padding-bottom: 10px; color: var(--accent); }
        .mode-toggle { display: flex; background: var(--bg); padding: 4px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #334155; gap: 4px; flex-wrap: wrap; }
        .mode-btn { flex: 1; padding: 8px; border: none; background: transparent; color: var(--text-mut); border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 0.8rem; }
        .mode-btn.active { background: var(--accent); color: var(--bg); }
        .chat-area { flex-grow: 1; overflow-y: auto; height: 320px; background: var(--bg); border-radius: 8px; padding: 15px; border: 1px solid #334155; margin-bottom: 15px; display: flex; flex-direction: column; gap: 10px; }
        .msg { padding: 8px 12px; border-radius: 8px; max-width: 85%; font-size: 0.9rem; line-height: 1.4; }
        .msg.user { background: #334155; align-self: flex-end; }
        .msg.ai { background: #0284c7; align-self: flex-start; }
        .input-box { display: flex; gap: 10px; }
        input[type="text"] { flex-grow: 1; background: var(--bg); border: 1px solid #334155; border-radius: 8px; padding: 12px; color: var(--text); outline: none; }
        button.send { background: var(--accent); color: var(--bg); border: none; padding: 12px 20px; border-radius: 8px; font-weight: bold; cursor: pointer; }
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .stat-card { background: var(--bg); border: 1px solid #334155; padding: 15px; border-radius: 8px; text-align: center; }
        .stat-val { font-size: 1.6rem; font-weight: bold; color: var(--accent); }
        .upload-zone { background: var(--bg); border: 2px dashed #334155; border-radius: 8px; padding: 20px; text-align: center; cursor: pointer; margin-top: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <!-- OPERATIONAL PANEL -->
        <div class="panel">
            <h2>👶 Child AI Operational Hub</h2>
            <div class="mode-toggle">
                <button id="btn-ANSWER" class="mode-btn active" onclick="setMode('ANSWER')">💬 ANSWER CONSOLE</button>
                <button id="btn-LEARN" class="mode-btn" onclick="setMode('LEARN')">🧠 DIRECT LESSON</button>
                <button id="btn-RAG" class="mode-btn" onclick="setMode('RAG')">📄 PDF RETRIEVAL</button>
                <button id="btn-WEB" class="mode-btn" onclick="setMode('WEB')">🌐 LIVE WEB SEARCH</button>
            </div>
            <div class="chat-area" id="chat-output">
                <div class="msg ai">System configured. I am a blank canvas. Teach me single terms, upload structural files, or route queries onto the live web index.</div>
            </div>
            <div class="input-box">
                <input type="text" id="chat-input" placeholder="Type message..." onkeypress="handleKey(event)">
                <button class="send" onclick="submitMessage()">Process</button>
            </div>
        </div>

        <!-- MATRIX CONTROL AND UPLOADS -->
        <div class="panel">
            <h2>📊 Active System Metrics</h2>
            <div class="stat-grid">
                <div class="stat-card">
                    <div>Synaptic Loss Deviation</div>
                    <div class="stat-val" id="telemetry-loss">0.0000</div>
                </div>
                <div class="stat-card">
                    <div>Data Nodes Indexed</div>
                    <div class="stat-val" id="telemetry-tokens">0</div>
                </div>
            </div>
            
            <h2>📂 Document Upload Engine</h2>
            <div class="upload-zone" onclick="document.getElementById('file-input').click()">
                <div style="color: var(--accent); font-size: 1.5rem; margin-bottom: 5px;">📤</div>
                <div id="upload-status" style="font-size: 0.85rem; color: var(--text-mut);">Click to select and index a PDF file into local memory matrices</div>
                <input type="file" id="file-input" style="display:none" accept=".pdf" onchange="uploadPDF()">
            </div>
        </div>
    </div>

    <script>
        let currentMode = "ANSWER";

        function setMode(mode) {
            currentMode = mode;
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('btn-' + mode).classList.add('active');
            
            let tip = "Type a message...";
            if (mode === "LEARN") tip = "Format: context word -> expected reply target";
            if (mode === "RAG") tip = "Ask a question about the facts inside the uploaded PDF document...";
            if (mode === "WEB") tip = "Enter search queries to scrape the web index natively...";
            document.getElementById('chat-input').placeholder = tip;
        }

        function handleKey(e) {
            if (e.key === 'Enter') submitMessage();
        }

        function submitMessage() {
            const inputEl = document.getElementById('chat-input');
            const text = inputEl.value.trim();
            if (!text) return;
            appendMsg(text, 'user');
            inputEl.value = '';
            
            fetch('/interact', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: currentMode, payload: text })
            })
            .then(res => res.json())
            .then(data => {
                if (data.reply) appendMsg(data.reply, 'ai');
                document.getElementById('telemetry-loss').innerText = data.loss.toFixed(4);
                document.getElementById('telemetry-tokens').innerText = data.tokens;
            });
        }

        function uploadPDF() {
            const fileInput = document.getElementById('file-input');
            if (fileInput.files.length === 0) return;
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            document.getElementById('upload-status').innerText = "⏳ Processing matrix calculations... please wait.";
            
            fetch('/upload', { method: 'POST', body: formData })
            .then(res => res.json())
            .then(data => {
                document.getElementById('upload-status').innerText = data.status;
                document.getElementById('telemetry-tokens').innerText = data.tokens;
                appendMsg("System notification: " + data.status, 'ai');
            });
        }

        function appendMsg(text, sender) {
            const chatArea = document.getElementById('chat-output');
            const msgNode = document.createElement('div');
            msgNode.className = `msg ${sender}`;
            msgNode.innerText = text;
            chatArea.appendChild(msgNode);
            chatArea.scrollTop = chatArea.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_UI)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"status": "No file chunk mapped.", "tokens": ai_child.total_tokens_processed})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "Blank file context.", "tokens": ai_child.total_tokens_processed})
    
    save_path = "temp_uploaded.pdf"
    file.save(save_path)
    success = ai_child.digest_pdf(save_path)
    if os.path.exists(save_path):
        os.remove(save_path)
        
    msg = "🏆 PDF successfully parsed and matrix indexed!" if success else "❌ Error: Extraction metrics returned zero data strings."
    return jsonify({"status": msg, "tokens": ai_child.total_tokens_processed})

@app.route('/interact', methods=['POST'])
def interact():
    data = request.json
    mode = data.get('mode')
    payload = data.get('payload', '').strip()
    current_loss = ai_child.loss_history[-1]
    
    if mode == "LEARN":
        if "->" in payload:
            input_side, target_side = payload.split("->", 1)
            current_loss = ai_child.add_conversational_memory(input_side, target_side)
            reply_msg = f"🧠 Memory Linked! Saved '{input_side.strip()}' directly to retrieve '{target_side.strip()}'."
        else:
            reply_msg = "⚠️ Formatting error! Use: word -> reply"
    elif mode == "RAG":
        reply_msg = f"📄 Extracted PDF Context Match:\n\n{ai_child.query_pdf_vault(payload)}"
    elif mode == "WEB":
        reply_msg = f"🌐 Scraped Live Internet Summary:\n\n{ai_child.search_live_internet(payload)}"
    else:
        # Standard fallback conversational matching lookups
        lookup = payload.lower()
        reply_msg = f"🤖 Internal Memory Response: '{ai_child.chat_memory_vault[lookup]}'" if lookup in ai_child.chat_memory_vault else "🤖 Internal Memory Response: 'Unmapped word vector sequence detected.'"
        
    return jsonify({
        "reply": reply_msg,
        "loss": current_loss,
        "tokens": ai_child.total_tokens_processed
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)