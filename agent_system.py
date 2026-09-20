import torch
import json
import os
import random
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader

app = Flask(__name__)
CORS(app)

# -------------------------------------------------------------
# 🧠 THE EVOLVING AI AGENT CORE WITH HUMAN LAYER
# -------------------------------------------------------------
class HumanLikeBlankAgent:
    def __init__(self):
        # Starts with an empty mind, but records everything
        self.chat_history_vault = []   # [{ "user": "...", "agent": "..." }] -> Long-term memory!
        self.rag_document_chunks = []  # Document memory vault
        self.total_nodes_indexed = 0
        
        # 🎭 NATIVE HUMAN PERSONALITY FILTER TEMPLATES
        self.human_intros = [
            "Oh, based on what we talked about earlier, ",
            "If I remember correctly from our chat, ",
            "Right, looking through my notes, ",
            "Ah, yes! ",
            "From what you taught me, "
        ]
        self.human_outros = [
            " Does that sound right?",
            " Hopefully that connects the dots!",
            " Let me know if I should remember this differently.",
            " Tell me if you want to add more to this."
        ]

    def learn_from_chat(self, user_phrase: str, agent_reply: str):
        """LEARNING MODE: Commits conversation patterns directly into long-term tracking tissue."""
        clean_user = user_phrase.strip().lower()
        clean_reply = agent_reply.strip()
        
        # Save to long-term memory logs
        self.chat_history_vault.append({"user": clean_user, "reply": clean_reply})
        self.total_nodes_indexed += len(clean_user.split()) + len(clean_reply.split())
        return f"✨ Memory committed! I will now remember that when you talk about '{clean_user}', I should think about '{clean_reply}'."

    def ingest_pdf_to_rag(self, file_path: str):
        """RAG LOADING MODE: Populates the document matrix."""
        try:
            reader = PdfReader(file_path)
            extracted_sentences = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 8]
                    extracted_sentences.extend(sentences)
            if not extracted_sentences:
                return "⚠️ Document parsing returned an empty signature."
            self.rag_document_chunks.extend(extracted_sentences)
            self.total_nodes_indexed += len(extracted_sentences)
            return f"🏆 Fed {len(extracted_sentences)} text vectors into my RAG database."
        except Exception as e:
            return f"❌ PDF Processing Error: {str(e)}"

    def query_long_term_memories(self, user_query: str) -> str:
        """🧠 MEMORY SCANNER: Scans ALL historical chats using keyword overlap loops."""
        if not self.chat_history_vault:
            return None
            
        clean_query = user_query.lower().strip()
        query_words = set(clean_query.split())
        
        best_memory_match = None
        max_overlap = 0
        
        # Scan backward from newest memories to oldest memories
        for memory in reversed(self.chat_history_vault):
            memory_words = set(memory["user"].split())
            overlap = len(query_words.intersection(memory_words))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_memory_match = memory["reply"]
                
        if max_overlap > 0:
            return best_memory_match
        return None

    def query_document_rag(self, user_query: str) -> str:
        """📄 RAG SCANNER: Scans uploaded text files."""
        if not self.rag_document_chunks:
            return None
        clean_query = user_query.lower().strip()
        query_words = set(clean_query.split())
        best_match = None
        max_overlap = 0
        for chunk in self.rag_document_chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = chunk
        if max_overlap > 0:
            return best_match
        return None

    def generate_human_response(self, raw_knowledge: str, type_source: str) -> str:
        """🎭 HUMAN LAYER ENGINE: Takes raw facts and wraps them in natural conversational styling."""
        intro = random.choice(self.human_intros)
        outro = random.choice(self.human_outros)
        
        if type_source == "CHAT_MEMORY":
            return f"{intro}you taught me: '{raw_knowledge}'.{outro}"
        elif type_source == "RAG_PDF":
            return f"Looking at the document you gave me, it says: \"{raw_knowledge}\". Hope that answers your question!"
        else:
            return "Hmm, my mind is completely empty on that topic right now. Could you teach me what that means, or upload a document about it?"

# Initialize the Evolving Humanized Agent
agent = HumanLikeBlankAgent()

# -------------------------------------------------------------
# 🌐 THE RESPONSIVE FRONTEND CONSOLE
# -------------------------------------------------------------
HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evolving Human-Like AI Agent</title>
    <style>
        :root { --bg: #090d16; --panel: #131a26; --accent: #38bdf8; --text: #f8fafc; --text-mut: #64748b; }
        body { font-family: sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; display: flex; justify-content: center; }
        .container { width: 100%; max-width: 1000px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .panel { background: var(--panel); border-radius: 12px; padding: 20px; border: 1px solid #1e293b; display: flex; flex-direction: column; }
        h2 { margin-top: 0; font-size: 1.2rem; color: var(--accent); border-bottom: 1px solid #1e293b; padding-bottom: 10px; }
        .tabs { display: flex; background: var(--bg); padding: 4px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #1e293b; }
        .tab-btn { flex: 1; padding: 10px; border: none; background: transparent; color: var(--text-mut); border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 0.85rem; }
        .tab-btn.active { background: var(--accent); color: var(--bg); }
        .display-window { flex-grow: 1; overflow-y: auto; height: 340px; background: var(--bg); border-radius: 8px; padding: 15px; border: 1px solid #1e293b; margin-bottom: 15px; display: flex; flex-direction: column; gap: 10px; }
        .msg { padding: 10px 14px; border-radius: 8px; max-width: 85%; font-size: 0.9rem; line-height: 1.4; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .msg.user { background: #1e293b; align-self: flex-end; border-bottom-right-radius: 2px; }
        .msg.agent { background: #0284c7; align-self: flex-start; border-bottom-left-radius: 2px; }
        .controls { display: flex; gap: 10px; }
        input[type="text"] { flex-grow: 1; background: var(--bg); border: 1px solid #1e293b; border-radius: 8px; padding: 12px; color: var(--text); outline: none; }
        button.action-btn { background: var(--accent); color: var(--bg); border: none; padding: 12px 20px; border-radius: 8px; font-weight: bold; cursor: pointer; }
        .metric-card { background: var(--bg); border: 1px solid #1e293b; padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 20px; }
        .metric-value { font-size: 2rem; font-weight: bold; color: var(--accent); }
        .upload-box { border: 2px dashed #1e293b; background: var(--bg); border-radius: 8px; padding: 30px; text-align: center; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <!-- CONSOLE TERMINAL -->
        <div class="panel">
            <h2>🤖 Natural Evolving Agent Terminal</h2>
            <div class="tabs">
                <button id="tab-CHAT" class="tab-btn active" onclick="switchMode('CHAT')">💬 TALK & EVALUATE</button>
                <button id="tab-TRAIN" class="tab-btn" onclick="switchMode('TRAIN')">🧠 TEACH NEW FACT</button>
            </div>
            <div class="display-window" id="console-logs">
                <div class="msg agent">Hello! I am completely empty right now, but I have long-term tracking activated. Whatever we talk about, or whatever PDF you upload, I will process it and adapt my speech style to answer you like a real peer. Try teaching me something!</div>
            </div>
            <div class="controls">
                <input type="text" id="console-input" placeholder="Type a message..." onkeypress="checkEnter(event)">
                <button class="action-btn" onclick="sendMessage()">Send</button>
            </div>
        </div>

        <!-- CONTROL METRICS PANEL -->
        <div class="panel">
            <h2>📊 Live Evolving Telemetry</h2>
            <div class="metric-card">
                <div>Total Synaptic Blocks Programmed</div>
                <div class="metric-value" id="nodes-counter">0</div>
            </div>
            
            <h2>📂 Document RAG Upload Pipeline</h2>
            <div class="upload-box" onclick="document.getElementById('file-picker').click()">
                <div style="font-size: 2rem; margin-bottom: 5px;">📄</div>
                <div id="upload-status" style="font-size: 0.85rem; color: var(--text-mut);">Index a PDF into my blank memory vault</div>
                <input type="file" id="file-picker" style="display:none" accept=".pdf" onchange="uploadPDFToServer()">
            </div>
        </div>
    </div>

    <script>
        let currentMode = "CHAT";

        function switchMode(mode) {
            currentMode = mode;
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById('tab-' + mode).classList.add('active');
            let hint = "Chat naturally... I will search my database and previous conversations!";
            if (mode === "TRAIN") hint = "Format fact: concept -> response data (e.g. your name -> Leo)";
            document.getElementById('console-input').placeholder = hint;
        }

        function checkEnter(e) { 
            if (e.key === 'Enter') sendMessage(); 
        }

        function sendMessage() {
            const inputField = document.getElementById('console-input');
            const dataText = inputField.value.trim();
            if(!dataText) return;

            appendLog(dataText, 'user');
            inputField.value = '';

            fetch('/interact', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: currentMode, payload: dataText })
            })
            .then(res => res.json())
            .then(data => {
                if(data.output) appendLog(data.output, 'agent');
                document.getElementById('nodes-counter').innerText = data.tokens;
            });
        }

        function uploadPDFToServer() {
            const picker = document.getElementById('file-picker');
            if(picker.files.length === 0) return;
            
            const dataObj = new FormData();
            dataObj.append('file', picker.files[0]);
            document.getElementById('upload-status').innerText = "⏳ Extracting text nodes into blank tracking vector list...";

            fetch('/upload', { method: 'POST', body: dataObj })
            .then(res => res.json())
            .then(data => {
                document.getElementById('upload-status').innerText = data.message;
                document.getElementById('nodes-counter').innerText = data.tokens;
                appendLog("System Update: " + data.message, 'agent');
            });
        }

        function appendLog(text, role) {
            const box = document.getElementById('console-logs');
            const div = document.createElement('div');
            div.className = `msg ${role}`;
            div.innerText = text;
            box.appendChild(div);
            box.scrollTop = box.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_DASHBOARD)

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({"message": "File processing error.", "tokens": agent.total_nodes_indexed})
    file = request.files['file']
    save_name = "active_rag_cache.pdf"
    file.save(save_name)
    response_msg = agent.ingest_pdf_to_rag(save_name)
    if os.path.exists(save_name):
        os.remove(save_name)
    return jsonify({"message": response_msg, "tokens": agent.total_nodes_indexed})

@app.route('/interact', methods=['POST'])
def handle_interaction():
    data = request.json
    mode = data.get('mode')
    payload = data.get('payload', '').strip()
    
    if mode == "TRAIN":
        if "->" in payload:
            phrase, reply = payload.split("->", 1)
            agent_response = agent.learn_from_chat(phrase, reply)
        else:
            agent_response = "⚠️ Invalid configuration format! Please use: keyword phrase -> what I should remember"
    else:  # 💬 CHAT & EVALUATE CONSOLE (Dynamic Multi-Route Scanner)
        # Step A: First cross-examine our long-term chat memory history logs
        historical_chat_match = agent.query_long_term_memories(payload)
        if historical_chat_match:
            agent_response = agent.generate_human_response(historical_chat_match, "CHAT_MEMORY")
        else:
            # Step B: Fall back to document context blocks if no chat matches exist
            document_rag_match = agent.query_document_rag(payload)
            if document_rag_match:
                agent_response = agent.generate_human_response(document_rag_match, "RAG_PDF")
            else:
                # Step C: Fall back to default empty mind trigger if nothing is found
                agent_response = agent.generate_human_response(None, "EMPTY")
                
        # Automatically log this ongoing conversation into long-term history tissue as well!
        agent.learn_from_chat(payload, agent_response)
        
    return jsonify({"output": agent_response, "tokens": agent.total_nodes_indexed})

if __name__ == '__main__':
    print("🚀 Launching Upgraded Humanized Agent Dashboard on port 5002...")
    app.run(debug=True, port=5002)