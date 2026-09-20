import torch
import json
from flask import Flask, render_template_string, request, jsonify
from brain_tissue import build_local_child_brain
from senses import CognitiveSenses
from correction import ParentalCorrection
from syntax_memory import RecurrentSyntaxLayer

app = Flask(__name__)

class ProductionAIChildSystem:
    def __init__(self):
        self.vocab = {
            "<PAD>": 0, "hello": 1, "hi": 2, "please": 3, "thanks": 4, 
            "be": 5, "polite": 6, "good": 7, "bad": 8, "stop": 9, "alert": 10
        }
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        self.architecture = [self.vocab_size, 8, self.vocab_size]
        self.brain = build_local_child_brain(self.architecture)
        self.recurrent_layer = RecurrentSyntaxLayer(input_size=self.vocab_size, hidden_size=8)
        
        self.loss_history = []
        self.total_tokens_processed = 0

    def sentence_to_tensor(self, sentence: str):
        words = [w.lower() for w in sentence.strip().split() if w.lower() in self.vocab]
        if not words:
            words = ["hello"]
            
        tensor = torch.zeros(len(words), self.vocab_size)
        for idx, word in enumerate(words):
            tensor[idx, self.vocab[word]] = 1.0
        return tensor, words

    def run_inference(self, sentence_tensor: torch.Tensor) -> torch.Tensor:
        thought_history = self.recurrent_layer.process_sequence(sentence_tensor)
        final_thought = thought_history[-1]
        raw_output_signal = torch.matmul(final_thought, self.brain.weights.t()) + self.brain.biases
        return torch.relu(raw_output_signal)

    def learn_behavioral_rule(self, input_phrase: str, target_word: str):
        x_tensor, words_list = self.sentence_to_tensor(input_phrase)
        y_target = torch.zeros(1, self.vocab_size)
        target_clean = target_word.strip().lower()
        
        if target_clean in self.vocab:
            y_target[0, self.vocab[target_clean]] = 1.0
        else:
            y_target[0, self.vocab["good"]] = 1.0

        epochs = 20
        last_loss = 0.0
        for _ in range(epochs):
            prediction = self.run_inference(x_tensor)
            loss = ParentalCorrection.calculate_error(prediction, y_target)
            loss.backward()
            ParentalCorrection.nudge_memory(self.brain, learning_rate=0.15)
            with torch.no_grad():
                if self.recurrent_layer.W_input.grad is not None:
                    self.recurrent_layer.W_input -= 0.15 * self.recurrent_layer.W_input.grad
                    self.recurrent_layer.W_recurrent -= 0.15 * self.recurrent_layer.W_recurrent.grad
                    self.recurrent_layer.bias -= 0.15 * self.recurrent_layer.bias.grad
                    self.recurrent_layer.W_input.grad.zero_()
                    self.recurrent_layer.W_recurrent.grad.zero_()
                    self.recurrent_layer.bias.grad.zero_()
            last_loss = loss.item()
            
        self.total_tokens_processed += len(words_list)
        self.loss_history.append(last_loss)
        return last_loss

    def evaluate_phrase(self, input_phrase: str) -> str:
        x_tensor, _ = self.sentence_to_tensor(input_phrase)
        with torch.no_grad():
            prediction = self.run_inference(x_tensor)
        predicted_idx = torch.argmax(prediction).item()
        return self.inverse_vocab.get(predicted_idx, "unusable output")

ai_child = ProductionAIChildSystem()

HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tabula Rasa AI Dashboard</title>
    <style>
        :root { --bg: #0f172a; --panel: #1e293b; --accent: #38bdf8; --text: #f8fafc; --text-mut: #94a3b8; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; display: flex; justify-content: center; }
        .container { width: 100%; max-width: 1000px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .panel { background: var(--panel); border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); border: 1px solid #334155; display: flex; flex-direction: column; }
        h2 { margin-top: 0; font-size: 1.25rem; border-bottom: 2px solid #334155; padding-bottom: 10px; display: flex; align-items: center; gap: 8px; }
        .mode-toggle { display: flex; background: var(--bg); padding: 4px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #334155; }
        .mode-btn { flex: 1; padding: 10px; border: none; background: transparent; color: var(--text-mut); border-radius: 6px; cursor: pointer; font-weight: bold; }
        .mode-btn.active { background: var(--accent); color: var(--bg); }
        .chat-area { flex-grow: 1; overflow-y: auto; height: 300px; background: var(--bg); border-radius: 8px; padding: 15px; border: 1px solid #334155; margin-bottom: 15px; display: flex; flex-direction: column; gap: 10px; }
        .msg { padding: 8px 12px; border-radius: 8px; max-width: 80%; font-size: 0.95rem; line-height: 1.4; }
        .msg.user { background: #334155; align-self: flex-end; color: var(--text); }
        .msg.ai { background: #0284c7; align-self: flex-start; color: white; }
        .input-box { display: flex; gap: 10px; }
        input[type="text"] { flex-grow: 1; background: var(--bg); border: 1px solid #334155; border-radius: 8px; padding: 12px; color: var(--text); outline: none; }
        input[type="text"]:focus { border-color: var(--accent); }
        button.send { background: var(--accent); color: var(--bg); border: none; padding: 12px 20px; border-radius: 8px; font-weight: bold; cursor: pointer; }
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px; }
        .stat-card { background: var(--bg); border: 1px solid #334155; padding: 15px; border-radius: 8px; text-align: center; }
        .stat-val { font-size: 1.8rem; font-weight: bold; color: var(--accent); margin-top: 5px; }
        .vocab-list { display: flex; flex-wrap: wrap; gap: 6px; background: var(--bg); border-radius: 8px; border: 1px solid #334155; max-height: 150px; overflow-y: auto; padding: 10px; }
        .vocab-tag { background: #1e293b; border: 1px solid #475569; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; color: var(--text-mut); }
    </style>
</head>
<body>
    <div class="container">
        <div class="panel">
            <h2>👶 Tabula Rasa Interface</h2>
            <div class="mode-toggle">
                <button id="btn-answer" class="mode-btn active" onclick="setMode('ANSWER')">💬 ANSWERING MODE</button>
                <button id="btn-learn" class="mode-btn" onclick="setMode('LEARN')">🧠 TRAINING MODE</button>
            </div>
            <div class="chat-area" id="chat-output">
                <div class="msg ai">Hello. My memory architecture is currently blank. Switch to Training Mode to build my synaptic pathways, or send a word to test me.</div>
            </div>
            <div class="input-box">
                <input type="text" id="chat-input" placeholder="Type pattern like 'please be polite -> polite'..." onkeypress="handleKey(event)">
                <button class="send" onclick="submitMessage()">Send</button>
            </div>
        </div>

        <div class="panel">
            <h2>📊 Real-Time Matrix Telemetry</h2>
            <div class="stat-grid">
                <div class="stat-card">
                    <div style="font-size:0.85rem; color:var(--text-mut)">Current Loss Error</div>
                    <div class="stat-val" id="telemetry-loss">0.0000</div>
                </div>
                <div class="stat-card">
                    <div style="font-size:0.85rem; color:var(--text-mut)">Synapses Programmed</div>
                    <div class="stat-val" id="telemetry-tokens">0</div>
                </div>
            </div>
            <h2>📚 Recognized Word Embeddings Matrix</h2>
            <div class="vocab-list" id="vocab-container"></div>
        </div>
    </div>

    <script id="vocab-data" type="application/json">VOCAB_PLACEHOLDER</script>
    <script>
        let currentMode = "ANSWER";
        const vocab = JSON.parse(document.getElementById('vocab-data').textContent);

        const container = document.getElementById('vocab-container');
        vocab.forEach(function(word) {
            if(word !== "<PAD>") {
                const tag = document.createElement('span');
                tag.className = 'vocab-tag';
                tag.innerText = word;
                container.appendChild(tag);
            }
        });

        function setMode(mode) {
            currentMode = mode;
            document.getElementById('btn-answer').classList.toggle('active', mode === 'ANSWER');
            document.getElementById('btn-learn').classList.toggle('active', mode === 'LEARN');
            
            const placeholderText = mode === 'ANSWER' 
                ? "Send words to evaluate model memory outputs..." 
                : "Enter training rules in format: sentence -> expected_reply_word";
            document.getElementById('chat-input').placeholder = placeholderText;
        }

        function handleKey(e) { 
            if (e.key === 'Enter') submitMessage(); 
        }

        function submitMessage() {
            const inputEl = document.getElementById('chat-input');
            const text = inputEl.value.trim();
            if(!text) return;

            appendMsg(text, 'user');
            inputEl.value = '';

            fetch('/interact', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: currentMode, payload: text })
            })
            .then(res => res.json())
            .then(data => {
                if(data.reply) appendMsg(data.reply, 'ai');
                document.getElementById('telemetry-loss').innerText = data.loss.toFixed(4);
                document.getElementById('telemetry-tokens').innerText = data.tokens;
            });
        }

        function appendMsg(text, sender) {
            const chatArea = document.getElementById('chat-output');
            const msgNode = document.createElement('div');
            msgNode.className = 'msg ' + sender;
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
    vocab_list = list(ai_child.vocab.keys())
    rendered_ui = HTML_UI.replace("VOCAB_PLACEHOLDER", json.dumps(vocab_list))
    return render_template_string(rendered_ui)

@app.route('/interact', methods=['POST'])
def interact():
    data = request.json
    mode = data.get('mode')
    payload = data.get('payload', '')
    current_loss = ai_child.loss_history[-1] if ai_child.loss_history else 0.0
    
    if mode == "LEARN":
        if "->" in payload:
            input_side, target_side = payload.split("->", 1)
            current_loss = ai_child.learn_behavioral_rule(input_side.strip(), target_side.strip())
            reply_msg = f"📝 Calculus adjustment triggered. Synaptic paths updated to connect phrase matching '{input_side.strip()}' with response target '{target_side.strip()}'."
        else:
            reply_msg = "⚠️ Parsing syntax error! Please use the explicit formatting: context words -> expected_response_word"
    else:
        ai_response = ai_child.evaluate_phrase(payload)
        reply_msg = f"🤖 Decoded Matrix Match: '{ai_response}'"
        
    return jsonify({
        "reply": reply_msg,
        "loss": current_loss,
        "tokens": ai_child.total_tokens_processed
    })

if __name__ == '__main__':
    print("🚀 Mounting production UI wrapper server on local network...")
    app.run(debug=True, port=5001)