import os
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader

app = Flask(__name__)
# Enable CORS so your free GitHub Pages frontend can securely pass signals here
CORS(app)

class HumanLikeBlankAgent:
    def __init__(self):
        self.chat_history_vault = []   
        self.rag_document_chunks = []  
        self.total_nodes_indexed = 0
        
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
        clean_user = user_phrase.strip().lower()
        clean_reply = agent_reply.strip()
        self.chat_history_vault.append({"user": clean_user, "reply": clean_reply})
        self.total_nodes_indexed += len(clean_user.split()) + len(clean_reply.split())
        return f"✨ Memory committed! I will now remember that when you talk about '{clean_user}', I should think about '{clean_reply}'."

    def ingest_pdf_to_rag(self, file_path: str):
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
        if not self.chat_history_vault:
            return None
        clean_query = user_query.lower().strip()
        query_words = set(clean_query.split())
        best_memory_match = None
        max_overlap = 0
        for memory in reversed(self.chat_history_vault):
            memory_words = set(memory["user"].split())
            overlap = len(query_words.intersection(memory_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_memory_match = memory["reply"]
        return best_memory_match if max_overlap > 0 else None

    def query_document_rag(self, user_query: str) -> str:
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
        return best_match if max_overlap > 0 else None

    def generate_human_response(self, raw_knowledge: str, type_source: str) -> str:
        intro = random.choice(self.human_intros)
        outro = random.choice(self.human_outros)
        if type_source == "CHAT_MEMORY":
            return f"{intro}you taught me: '{raw_knowledge}'.{outro}"
        elif type_source == "RAG_PDF":
            return f"Looking at the document you gave me, it says: \"{raw_knowledge}\". Hope that answers your question!"
        else:
            return "Hmm, my mind is completely empty on that topic right now. Could you teach me what that means, or upload a document about it?"

agent = HumanLikeBlankAgent()

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({"message": "File processing error.", "tokens": agent.total_nodes_indexed})
    file = request.files['file']
    save_name = "active_rag_cache.pdf"
    file.save(save_name)
    response_msg = agent.ingest_pdf_to_rag(save_name)
    if os.path.exists(save_name): os.remove(save_name)
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
            agent_response = "⚠️ Invalid configuration format! Use: keyword phrase -> what I should remember"
    else:
        historical_chat_match = agent.query_long_term_memories(payload)
        if historical_chat_match:
            agent_response = agent.generate_human_response(historical_chat_match, "CHAT_MEMORY")
        else:
            document_rag_match = agent.query_document_rag(payload)
            if document_rag_match:
                agent_response = agent.generate_human_response(document_rag_match, "RAG_PDF")
            else:
                agent_response = agent.generate_human_response(None, "EMPTY")
        agent.learn_from_chat(payload, agent_response)

    return jsonify({"output": agent_response, "tokens": agent.total_nodes_indexed})

if __name__ == '__main__':
    # Dynamic port configuration matching for Render cloud requirements
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
