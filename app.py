import torch
import json
import requests
import os
import gradio as gr
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

# -------------------------------------------------------------
# 🧠 THE UPGRADED CORE ENVIRONMENT WITH MEMORY & RAG
# -------------------------------------------------------------
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
        if not file_path:
            return "❌ Error: No file provided."
        try:
            reader = PdfReader(file_path)
            text_chunks = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    chunks = [c.strip() for c in text.split('.') if len(c.strip()) > 10]
                    text_chunks.extend(chunks)
                    
            if not text_chunks:
                return "❌ Error: Extraction metrics returned zero data strings."
                
            self.pdf_vault.extend(text_chunks)
            embeddings_array = self.embedding_model.encode(self.pdf_vault, convert_to_tensor=True)
            self.pdf_embeddings = embeddings_array
            self.total_tokens_processed += len(text_chunks) * 5
            return f"🏆 PDF successfully parsed and matrix indexed! ({len(text_chunks)} sentences added)"
        except Exception as e:
            return f"❌ PDF Processing Error: {str(e)}"

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
                if snippets:
                    return "\n\n".join(snippets[:3])
                results = [a.text for a in soup.find_all('a', class_='result__snippet') if a.text]
                if results:
                    return "\n\n".join(results[:3])
            return "Web server reached, but no clean snippet summary could be parsed."
        except Exception as e:
            return f"Network routing issue: {str(e)}"

# Instantiate System
ai_child = TabulaRasaKnowledgeCore()

# -------------------------------------------------------------
# 🌐 THE FRONTEND INTERFACE DESIGN (GRADIO)
# -------------------------------------------------------------
def process_interaction(mode, message, training_target):
    if mode == "🧠 DIRECT LESSON":
        if not message or not training_target:
            return "⚠️ Formatting error! For Training Mode, provide both the Input Phrase and Target Output Word.", ai_child.total_tokens_processed
        res = ai_child.add_conversational_memory(message, training_target)
        return res, ai_child.total_tokens_processed
        
    elif mode == "📄 PDF RETRIEVAL":
        res = f"📄 Extracted PDF Context Match:\n\n{ai_child.query_pdf_vault(message)}"
        return res, ai_child.total_tokens_processed
        
    elif mode == "🌐 LIVE WEB SEARCH":
        res = f"🌐 Scraped Live Internet Summary:\n\n{ai_child.search_live_internet(message)}"
        return res, ai_child.total_tokens_processed
        
    else: # ANSWER CONSOLE
        lookup = message.strip().lower()
        if lookup in ai_child.chat_memory_vault:
            reply_msg = f"🤖 Internal Memory Response: '{ai_child.chat_memory_vault[lookup]}'"
        else:
            reply_msg = "🤖 Internal Memory Response: 'Unmapped word vector sequence detected.'"
        return reply_msg, ai_child.total_tokens_processed

def handle_pdf_upload(file):
    if file is None:
        return "No file selected.", ai_child.total_tokens_processed
    status_msg = ai_child.digest_pdf(file.name)
    return status_msg, ai_child.total_tokens_processed

# Create Gradio UI Layout
with gr.Blocks(theme=gr.themes.Soft(primary_hue="sky", neutral_hue="slate")) as demo:
    gr.Markdown("# 👶 Tabula Rasa Advanced RAG System")
    gr.Markdown("A completely free, blank-slate AI that learns dynamically from conversations, uploaded PDFs, or live web indexes.")
    
    with gr.Row():
        with gr.Column(scale=2):
            mode_selection = gr.Radio(
                choices=["💬 ANSWER CONSOLE", "🧠 DIRECT LESSON", "📄 PDF RETRIEVAL", "🌐 LIVE WEB SEARCH"],
                value="💬 ANSWER CONSOLE",
                label="System Mode Select"
            )
            
            chat_input = gr.Textbox(label="Message / Input Phrase / Search Query", placeholder="Type here...")
            target_input = gr.Textbox(label="Expected Reply Target (Used ONLY for Direct Lesson mode)", placeholder="e.g., polite", visible=False)
            
            # Show/hide target box based on mode choice
            def toggle_target_box(mode):
                return gr.update(visible=(mode == "🧠 DIRECT LESSON"))
            mode_selection.change(toggle_target_box, inputs=mode_selection, outputs=target_input)
            
            submit_btn = gr.Button("Process Matrix Signature", variant="primary")
            output_box = gr.Textbox(label="AI Child System Response Output", interactive=False)
            
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Active System Metrics")
            token_counter = gr.Number(label="Data Nodes Indexed", value=0, interactive=False)
            
            gr.Markdown("### 📂 Document Upload Engine")
            pdf_upload = gr.File(label="Upload PDF Document", file_types=[".pdf"])
            upload_status = gr.Textbox(label="Upload Tracking Metrics", interactive=False)
            
            pdf_upload.change(handle_pdf_upload, inputs=pdf_upload, outputs=[upload_status, token_counter])

    submit_btn.click(
        process_interaction, 
        inputs=[mode_selection, chat_input, target_input], 
        outputs=[output_box, token_counter]
    )

if __name__ == '__main__':
    demo.launch()
