import os
import sys
import json
import traceback
import datetime
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np


try:
    import gguf
    import numpy as _np

    def _gguf_get_buffer(reader):
        for attr in ("_mmap", "_buf", "_buffer", "buffer", "_data", "_mem"):
            if hasattr(reader, attr):
                return getattr(reader, attr)
        return None  

    if hasattr(gguf.gguf_reader.GGUFReader, "_get"):
        def _patched_get(self, offs, dtype, override_order=None):
            byte_order = override_order or getattr(self, 'byte_order', '<')
            dt = _np.dtype(dtype)
            nbytes = dt.itemsize
            buf = _gguf_get_buffer(self)

            if buf is not None:
                mv = memoryview(buf)[offs:offs+nbytes]
                arr = _np.frombuffer(mv, dtype=dt, count=1)
            else:
                f = getattr(self, "_f", None) or getattr(self, "f", None) or getattr(self, "_fp", None)
                if f is None:
                    raise AttributeError("GGUFReader internal buffer attribute not found and no file handle found")
                pos = f.tell()
                f.seek(offs)
                data = f.read(nbytes)
                f.seek(pos)
                arr = _np.frombuffer(data, dtype=dt, count=1)

            return arr.view(arr.dtype.newbyteorder(byte_order))

        gguf.gguf_reader.GGUFReader._get = _patched_get
except Exception:
    pass


from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt', quiet=True)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch  
except Exception:
    AutoTokenizer = None  
    AutoModelForCausalLM = None  
    torch = None  

try:
    import llama_cpp  
except Exception:
    llama_cpp = None  

try:
    import psutil
except ImportError:
    psutil = None

def neural_network_inference(prompt):
    return "Neural fallback response: " + prompt[::-1]

def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-12 else vec

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class QuantumLanguageModel:
    def __init__(self):
        self.vocab_size = None
        self.embed_dim = None
        self.hidden_dim = None
        self.embeddings = None
        self.W_out = None
        self.W_proj = None
        self.token_to_id = {}  
        self.id_to_token = {}  

    def load_from_file(self, model_file_path, token_map_file_path=None):
        if not os.path.isfile(model_file_path):
            raise FileNotFoundError(f"Model file '{model_file_path}' does not exist.")
        _, ext = os.path.splitext(model_file_path)
        if ext.lower() not in ['.json', '.qelm']:
            raise ValueError("Unsupported file format. Use a .json or .qelm file.")
        try:
            with open(model_file_path, 'r', encoding='utf-8') as f:
                model_dict = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load model file: {e}")
        if "version" in model_dict and model_dict["version"] == "4.0": #If this instance gives issues for pulling old models, then update or open a pr.
            self.vocab_size = model_dict["vocab_size"]
            self.embed_dim = model_dict["embed_dim"]
            self.hidden_dim = model_dict["hidden_dim"]
            self.embeddings = np.array(model_dict["embeddings"], dtype=np.float32)
            self.W_out = (np.array(model_dict["W_out"], dtype=np.float32)
                          if "W_out" in model_dict else
                          np.random.randn(self.vocab_size, self.embed_dim).astype(np.float32)*0.01)
            self.W_proj = (np.array(model_dict["W_proj"], dtype=np.float32)
                           if ("W_proj" in model_dict and model_dict["W_proj"] is not None)
                           else None)
            if "token_to_id" in model_dict and "id_to_token" in model_dict:
                self.token_to_id = model_dict["token_to_id"]
                self.id_to_token = {int(k): v for k, v in model_dict["id_to_token"].items()}
                if all(isinstance(k, str) and k.startswith("<TOKEN_") and k.endswith(">") for k in self.token_to_id.keys()):
                    self._generate_friendly_token_map()
            elif token_map_file_path and os.path.isfile(token_map_file_path):
                self.load_token_map_from_file(token_map_file_path)
            elif "vocabulary" in model_dict:
                tokens = model_dict["vocabulary"]
                self.token_to_id = {token: idx for idx, token in enumerate(tokens)}
                self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
            else:
                self.token_to_id = {}
                self.id_to_token = {}
        else:
            required = ["vocab_size", "embed_dim", "hidden_dim", "embeddings"]
            for key in required:
                if key not in model_dict:
                    raise KeyError(f"Missing required key '{key}' in model file.")
            self.vocab_size = model_dict["vocab_size"]
            self.embed_dim = model_dict["embed_dim"]
            self.hidden_dim = model_dict["hidden_dim"]
            self.embeddings = np.array(model_dict["embeddings"], dtype=np.float32)
            if "token_to_id" in model_dict and "id_to_token" in model_dict:
                self.token_to_id = model_dict["token_to_id"]
                self.id_to_token = {int(k): v for k, v in model_dict["id_to_token"].items()}
                if all(isinstance(k, str) and k.startswith("<TOKEN_") and k.endswith(">") for k in self.token_to_id.keys()):
                    self._generate_friendly_token_map()
            elif "vocabulary" in model_dict:
                tokens = model_dict["vocabulary"]
                self.token_to_id = {token: idx for idx, token in enumerate(tokens)}
                self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
            else:
                self.token_to_id = {}
                self.id_to_token = {}
            self.W_out = (np.array(model_dict["W_out"], dtype=np.float32)
                          if "W_out" in model_dict else
                          np.random.randn(self.vocab_size, self.embed_dim).astype(np.float32)*0.01)
            if "W_proj" in model_dict and model_dict["W_proj"] is not None:
                self.W_proj = np.array(model_dict["W_proj"], dtype=np.float32)
            else:
                self.W_proj = None

    def _generate_friendly_token_map(self):
        common_words = [
            "the", "of", "and", "to", "in", "a", "is", "that", "it", "was",
            "I", "for", "on", "you", "with", "as", "be", "at", "by", "he",
            "this", "had", "not", "are", "but", "his", "they", "from", "she", "which"
        ]
        new_token_to_id = {}
        new_id_to_token = {}
        for token, idx in self.token_to_id.items():
            if token.startswith("<TOKEN_") and token.endswith(">"):
                if idx < len(common_words):
                    new_token = common_words[idx]
                else:
                    new_token = f"word{idx}"
            else:
                new_token = token
            new_token_to_id[new_token] = idx
            new_id_to_token[idx] = new_token
        self.token_to_id = new_token_to_id
        self.id_to_token = new_id_to_token
        logging.info("Placeholder token mapping replaced with human-friendly tokens.")

    def load_token_map_from_file(self, token_map_file_path):
        try:
            with open(token_map_file_path, 'r', encoding='utf-8') as f:
                token_map = json.load(f)
            if "token_to_id" in token_map:
                self.token_to_id = token_map["token_to_id"]
                if "id_to_token" in token_map:
                    self.id_to_token = {int(k): v for k, v in token_map["id_to_token"].items()}
                else:
                    self.id_to_token = {v: int(k) for k, v in self.token_to_id.items()}
            elif "vocabulary" in token_map:
                tokens = token_map["vocabulary"]
                self.token_to_id = {token: idx for idx, token in enumerate(tokens)}
                self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
            elif isinstance(token_map, dict):
                if all(isinstance(k, str) and k.startswith("<TOKEN_") and k.endswith(">") for k in token_map.keys()):
                    self.token_to_id = token_map
                    self._generate_friendly_token_map()
                else:
                    self.token_to_id = token_map
                    self.id_to_token = {v: k for k, v in token_map.items()}
            else:
                raise ValueError("Token mapping file is invalid.")
        except Exception as e:
            raise ValueError(f"Error loading token mapping: {e}")

    def forward(self, input_ids: list, use_residual: bool = True):
        try:
            vec = np.sum(self.embeddings[input_ids], axis=0)
            if self.W_proj is not None:
                vec = self.W_proj @ vec
            logits = self.W_out @ vec
            return logits
        except Exception as e:
            raise ValueError(f"Error during forward pass: {e}")

    def run_inference(self, prompt, max_length=20, temperature=1.0, char_level=False):
        if self.embeddings is None or self.W_out is None or not self.token_to_id:
            return neural_network_inference(prompt)
        tokens = list(prompt.lower()) if char_level else word_tokenize(prompt.lower())
        if tokens:
            input_ids = [self.token_to_id.get(token, self.token_to_id.get("<UNK>", 0)) for token in tokens]
        else:
            input_ids = [0]
        response_tokens = []
        current_ids = input_ids.copy()
        for _ in range(max_length):
            logits = self.forward(current_ids, use_residual=True)
            probs = softmax(logits / max(1e-12, temperature))
            sampled_id = int(np.random.choice(self.vocab_size, p=probs))
            sampled_token = self.id_to_token.get(sampled_id, "<UNK>")
            response_tokens.append(sampled_token)
            current_ids = [sampled_id]
            if sampled_token in [".", "!", "?"]:
                break
        return "".join(response_tokens) if char_level else " ".join(response_tokens)

class ChatDisplay(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, background="#f0f0f0")
        self.frame = tk.Frame(self.canvas, background="#f0f0f0")
        self.vsb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")
        self.frame.bind("<Configure>", self.on_frame_configure)

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def clear(self):
        for widget in self.frame.winfo_children():
            widget.destroy()

    def add_message(self, sender, timestamp, message, font_size, align, show_timestamp):
        anchor_val = align if align in ["e", "w", "center"] else "center"
        justify_mapping = {"e": "right", "w": "left", "center": "center"}
        justify_val = justify_mapping.get(align, "center")
        bubble_frame = tk.Frame(self.frame, bg="#f0f0f0", padx=5, pady=5)
        sender_label = tk.Label(bubble_frame, text=sender, font=("Helvetica", font_size, "bold"),
                                fg="#1a73e8" if anchor_val=="e" else "#34a853", bg="#f0f0f0")
        sender_label.pack(anchor="center", padx=5)
        if anchor_val == "e":
            bubble_bg = "#e8f0fe"
        elif anchor_val == "w":
            bubble_bg = "#e6f4ea"
        else:
            bubble_bg = "#fce8e6"
        msg_label = tk.Label(bubble_frame, text=message, font=("Helvetica", font_size),
                             fg="#1a73e8" if anchor_val=="e" else "#34a853" if anchor_val=="w" else "#ea4335",
                             bg=bubble_bg, wraplength=600, justify=justify_val, padx=10, pady=5, bd=1, relief="ridge")
        msg_label.pack(anchor=anchor_val, padx=5, pady=(0,5))
        if show_timestamp:
            ts_label = tk.Label(bubble_frame, text=timestamp, font=("Helvetica", int(font_size*0.8)),
                                fg="#888888", bg="#f0f0f0")
            ts_label.pack(anchor="center", pady=(0,5))
        bubble_frame.pack(fill="x", pady=5, padx=10)
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

class QELMChatUI:
    def __init__(self, root):
        self.root = root
        self.root.title("QELM Chat")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        self.current_theme = "light"
        self.font_size_var = tk.IntVar(value=12)
        self.temperature_var = tk.DoubleVar(value=1.0)
        self.max_tokens_var = tk.IntVar(value=20)
        self.show_timestamps_var = tk.BooleanVar(value=True)
        self.char_level_var = tk.BooleanVar(value=False)
        self.ai_name_var = tk.StringVar(value="QELM")
        self.user_name_var = tk.StringVar(value="User")
        self.model = QuantumLanguageModel()
        self.conversations = {}
        self.current_convo = "Default"
        self.conversations[self.current_convo] = []
        self.last_model_path = None
        self.last_token_map_path = None
        self.model_type_var = tk.StringVar(value="QELM")
        self.hf_model = None
        self.hf_tokenizer = None
        self.hf_device = None
        self.local_model = None
        if psutil:
            self.process = psutil.Process(os.getpid())
            self.process.cpu_percent(interval=None)
        else:
            self.process = None
        self.create_ui()
        self.update_resource_usage()

    def create_ui(self):
        menu_bar = tk.Menu(self.root)
        file_menu = tk.Menu(menu_bar, tearoff=False)
        file_menu.add_command(label="Import Dataset", command=self.import_dataset)
        file_menu.add_command(label="Load Token Map", command=self.menu_load_token_map)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        help_menu = tk.Menu(menu_bar, tearoff=False)
        help_menu.add_command(label="About", command=self.show_about_info)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menu_bar)

        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        ttk.Label(top_frame, text="Model:", font=("Helvetica", 12, "bold")).pack(side=tk.LEFT, padx=(0,5))
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(top_frame, textvariable=self.model_var, width=40)
        self.model_combo['values'] = []
        self.model_combo.set("Select or Load a QELM (.json/.qelm) Model...")
        self.model_combo.pack(side=tk.LEFT, padx=5)
        load_model_btn = ttk.Button(top_frame, text="Load Model", command=self.load_model)
        load_model_btn.pack(side=tk.LEFT, padx=5)

        ttk.Label(top_frame, text="Type:", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(15,2))
        self.model_type_combo = ttk.Combobox(top_frame, textvariable=self.model_type_var,
                                             values=["QELM", "HuggingFace", "LocalGGUF"], state="readonly", width=12)
        self.model_type_combo.pack(side=tk.LEFT, padx=(0,5))
        hf_btn = ttk.Button(top_frame, text="Load HF", command=self.load_hf_model)
        hf_btn.pack(side=tk.LEFT, padx=(0,5))
        local_btn = ttk.Button(top_frame, text="Load Local", command=self.load_local_model)
        local_btn.pack(side=tk.LEFT, padx=(0,5))
        ttk.Label(top_frame, text="Font Size:", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(30,2))
        self.font_spin = ttk.Spinbox(top_frame, from_=8, to=32, textvariable=self.font_size_var, width=3, command=self.update_font_size)
        self.font_spin.pack(side=tk.LEFT, padx=(0,10))
        theme_btn = ttk.Button(top_frame, text="Toggle Theme", command=self.toggle_theme)
        theme_btn.pack(side=tk.RIGHT, padx=5)

        left_frame = ttk.Frame(self.root, width=250)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)
        ttk.Label(left_frame, text="Conversations", font=("Helvetica", 14, "bold")).pack(pady=5)
        self.convo_list = tk.Listbox(left_frame, font=("Helvetica", 12))
        self.convo_list.pack(fill=tk.BOTH, expand=True)
        self.convo_list.bind("<<ListboxSelect>>", self.switch_conversation)
        new_conv_btn = ttk.Button(left_frame, text="New Conversation", command=self.new_conversation)
        new_conv_btn.pack(pady=5)
        rename_conv_btn = ttk.Button(left_frame, text="Rename Conversation", command=self.rename_conversation)
        rename_conv_btn.pack(pady=5)
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill='x', pady=5)
        adv_options_frame = ttk.LabelFrame(left_frame, text="Advanced Options")
        adv_options_frame.pack(fill=tk.X, pady=5)
        ttk.Label(adv_options_frame, text="Max Tokens:", font=("Helvetica", 10)).grid(row=0, column=0, sticky='e', padx=5, pady=5)
        tk.Spinbox(adv_options_frame, from_=1, to=100, textvariable=self.max_tokens_var, width=5).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(adv_options_frame, text="Temperature:", font=("Helvetica", 10)).grid(row=1, column=0, sticky='e', padx=5, pady=5)
        tk.Spinbox(adv_options_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.temperature_var, width=5).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        ttk.Checkbutton(adv_options_frame, text="Show Timestamps", variable=self.show_timestamps_var).grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        ttk.Checkbutton(adv_options_frame, text="Character-level Encoding", variable=self.char_level_var).grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        ttk.Label(adv_options_frame, text="AI Name:", font=("Helvetica", 10)).grid(row=4, column=0, sticky='e', padx=5, pady=5)
        ai_name_entry = ttk.Entry(adv_options_frame, textvariable=self.ai_name_var, width=14)
        ai_name_entry.grid(row=4, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(adv_options_frame, text="User Name:", font=("Helvetica", 10)).grid(row=5, column=0, sticky='e', padx=5, pady=5)
        user_name_entry = ttk.Entry(adv_options_frame, textvariable=self.user_name_var, width=14)
        user_name_entry.grid(row=5, column=1, sticky='w', padx=5, pady=5)
        reload_btn = ttk.Button(adv_options_frame, text="Reload Model/Map", command=self.reload_model_token_map)
        reload_btn.grid(row=6, column=0, columnspan=2, padx=5, pady=5)
        resource_frame = ttk.LabelFrame(left_frame, text="Resources")
        resource_frame.pack(fill=tk.X, pady=5)
        self.cpu_usage_label = ttk.Label(resource_frame, text="CPU Usage: N/A", font=("Helvetica", 9))
        self.cpu_usage_label.pack(anchor="w", padx=5, pady=2)
        self.gpu_usage_label = ttk.Label(resource_frame, text="GPU Usage: N/A", font=("Helvetica", 9))
        self.gpu_usage_label.pack(anchor="w", padx=5, pady=2)

        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.chat_canvas = ChatDisplay(right_frame)
        self.chat_canvas.pack(fill=tk.BOTH, expand=True)
        bottom_frame = ttk.Frame(right_frame)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        self.entry_var = tk.StringVar()
        self.entry_field = ttk.Entry(bottom_frame, textvariable=self.entry_var, font=("Helvetica", 12))
        self.entry_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,5))
        self.entry_field.bind("<Return>", self.handle_send)
        send_btn = ttk.Button(bottom_frame, text="Send", command=self.handle_send)
        send_btn.pack(side=tk.LEFT, padx=(0,5))
        clear_btn = ttk.Button(bottom_frame, text="Clear Chat", command=self.clear_chat)
        clear_btn.pack(side=tk.LEFT, padx=(0,5))
        save_btn = ttk.Button(bottom_frame, text="Save Chat", command=self.save_chat)
        save_btn.pack(side=tk.LEFT, padx=(0,5))
        self.refresh_chat_display()
        self.refresh_convo_list()

    def import_dataset(self):
        file_path = filedialog.askopenfilename(title="Select Dataset File",
                                               filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            self.system_message(f"Dataset {os.path.basename(file_path)} imported (not actually used).")

    def menu_load_token_map(self):
        file_path = filedialog.askopenfilename(title="Load Token Map",
                                               filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if file_path:
            self.load_token_map_from_file(file_path)

    def show_about_info(self):
        messagebox.showinfo("About QELM Chat", "QELM Chat UI\nVersion 1.1\nBy R&D BioTech Alaska")

    def load_model(self):
        model_path = filedialog.askopenfilename(title="Select QELM Model File",
                                                filetypes=[("QELM Files", "*.json *.qelm"), ("All Files", "*.*")])
        if not model_path:
            return
        try:
            token_map = ""
            if messagebox.askyesno("Token Mapping", "Does this model require a separate token mapping file?"):
                token_map = filedialog.askopenfilename(title="Select Token Mapping File",
                                                       filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
            self.model.load_from_file(model_path, token_map)
            current = list(self.model_combo['values'])
            if model_path not in current:
                current.append(model_path)
                self.model_combo['values'] = current
            self.model_combo.set(model_path)
            self.system_message(f"Model loaded from {os.path.basename(model_path)}")
        except Exception as e:
            self.system_message(f"Model load failed: {e}")
            messagebox.showerror("Model Load Error", f"An error occurred while loading the model:\n{e}")

    def reload_model_token_map(self):
        if not self.last_model_path:
            messagebox.showinfo("Reload Error", "No previous model file to reload.")
            return
        try:
            self.model.load_from_file(self.last_model_path, self.last_token_map_path)
            self.system_message(f"Reloaded model: {os.path.basename(self.last_model_path)}")
        except Exception as e:
            self.system_message(f"Reload failed: {e}")
            messagebox.showerror("Reload Error", f"Reload error:\n{e}")

    def toggle_theme(self):
        if self.current_theme == "light":
            self.root.configure(bg="#2e2e2e")
            self.chat_canvas.canvas.configure(bg="#2e2e2e")
            self.current_theme = "dark"
        else:
            self.root.configure(bg="#f0f0f0")
            self.chat_canvas.canvas.configure(bg="#f0f0f0")
            self.current_theme = "light"

    def new_conversation(self):
        name = f"Conversation {len(self.conversations) + 1}"
        self.conversations[name] = []
        self.current_convo = name
        self.refresh_convo_list()
        self.refresh_chat_display()

    def rename_conversation(self):
        selection = self.convo_list.curselection()
        if not selection:
            messagebox.showinfo("Rename Conversation", "Select a conversation to rename.")
            return
        idx = selection[0]
        old_name = self.convo_list.get(idx)
        new_name = simpledialog.askstring("Rename Conversation", "Enter new conversation name:", initialvalue=old_name)
        if not new_name:
            return
        if new_name in self.conversations and new_name != old_name:
            messagebox.showwarning("Name Conflict", f"A conversation named '{new_name}' already exists.")
            return
        self.conversations[new_name] = self.conversations.pop(old_name)
        if self.current_convo == old_name:
            self.current_convo = new_name
        self.refresh_convo_list()
        self.refresh_chat_display()

    def switch_conversation(self, event):
        selection = self.convo_list.curselection()
        if selection:
            idx = selection[0]
            name = self.convo_list.get(idx)
            self.current_convo = name
            self.refresh_chat_display()

    def refresh_convo_list(self):
        self.convo_list.delete(0, tk.END)
        for name in self.conversations.keys():
            self.convo_list.insert(tk.END, name)

    def handle_send(self, event=None):
        user_text = self.entry_var.get().strip()
        if not user_text:
            return
        user_label = self.user_name_var.get().strip() or "User"
        ai_label = self.ai_name_var.get().strip() or "QELM"
        self.add_message(user_label, user_text)
        self.entry_var.set("")
        model_type = self.model_type_var.get() or "QELM"
        response = None
        if model_type == "HuggingFace":
            if self.hf_model and self.hf_tokenizer:
                try:
                    response = self.run_hf_inference(user_text)
                except Exception as e:
                    self.system_message(f"HF inference failed: {e}")
                    response = None
            else:
                self.system_message("HuggingFace model not loaded; falling back to QELM.")
        elif model_type == "LocalGGUF":
            if self.local_model:
                try:
                    response = self.run_local_inference(user_text)
                except Exception as e:
                    self.system_message(f"Local model inference failed: {e}")
                    response = None
            else:
                self.system_message("Local model not loaded; falling back to QELM.")
        if response is None:
            try:
                response = self.model.run_inference(
                    user_text,
                    max_length=self.max_tokens_var.get(),
                    temperature=self.temperature_var.get(),
                    char_level=self.char_level_var.get()
                )
            except Exception as e:
                err = f"QELM inference failed: {e}"
                self.system_message(err)
                use_nn = messagebox.askyesno("Inference Error", f"{err}\nUse neural network fallback?")
                if use_nn:
                    response = neural_network_inference(user_text)
                else:
                    response = "<Error: Response generation failed>"
        self.add_message(ai_label, response)

    def add_message(self, sender, text):
        timestamp = datetime.datetime.now().strftime("%H:%M")
        msg = (sender, timestamp, text)
        if self.current_convo not in self.conversations:
            self.conversations[self.current_convo] = []
        self.conversations[self.current_convo].append(msg)
        self.refresh_chat_display()

    def refresh_chat_display(self):
        self.chat_canvas.clear()
        font_size = self.font_size_var.get()
        show_ts = self.show_timestamps_var.get()
        for sender, timestamp, text in self.conversations.get(self.current_convo, []):
            user_label = self.user_name_var.get().strip() or "User"
            ai_label = self.ai_name_var.get().strip() or "QELM"
            if sender == user_label:
                align = "e" 
            elif sender == ai_label:
                align = "w" 
            else:
                align = "center"
            self.chat_canvas.add_message(sender, timestamp, text, font_size, align, show_ts)
    
    def system_message(self, text):
        self.add_message("System", text)

    def clear_chat(self):
        if messagebox.askyesno("Clear Chat", "Are you sure you want to clear this conversation?"):
            self.conversations[self.current_convo] = []
            self.refresh_chat_display()

    def save_chat(self):
        if not self.current_convo or self.current_convo not in self.conversations:
            messagebox.showinfo("Save Chat", "No conversation to save.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
                                                 title="Save Conversation")
        if not file_path:
            return
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"Conversation: {self.current_convo}\n")
                f.write(f"Saved on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for sender, timestamp, text in self.conversations[self.current_convo]:
                    f.write(f"[{timestamp}] {sender}:\n{text}\n\n")
            messagebox.showinfo("Save Chat", f"Conversation saved to {file_path}.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save conversation:\n{e}")

    def load_token_map_from_file(self, token_map_file_path):
        try:
            with open(token_map_file_path, 'r', encoding='utf-8') as f:
                token_map = json.load(f)
            if "token_to_id" in token_map:
                self.model.token_to_id = token_map["token_to_id"]
                if "id_to_token" in token_map:
                    self.model.id_to_token = {int(k): v for k, v in token_map["id_to_token"].items()}
                else:
                    self.model.id_to_token = {v: int(k) for k, v in self.model.token_to_id.items()}
            elif "vocabulary" in token_map:
                tokens = token_map["vocabulary"]
                self.model.token_to_id = {token: idx for idx, token in enumerate(tokens)}
                self.model.id_to_token = {idx: token for token, idx in self.model.token_to_id.items()}
            elif isinstance(token_map, dict):
                if all(isinstance(k, str) and k.startswith("<TOKEN_") and k.endswith(">") for k in token_map.keys()):
                    self.model.token_to_id = token_map
                    self.model._generate_friendly_token_map()
                else:
                    self.model.token_to_id = token_map
                    self.model.id_to_token = {v: k for k, v in token_map.items()}
            else:
                raise ValueError("Token mapping file is invalid.")
            self.system_message("Token map loaded successfully.")
        except Exception as e:
            self.system_message(f"Load token map error: {e}")
            messagebox.showerror("Load Error", f"Failed to load token map:\n{e}")

    def load_token_map(self):
        try:
            file_path = filedialog.askopenfilename(title="Load Token Map",
                                                   filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
            if file_path:
                self.load_token_map_from_file(file_path)
                self.display_token_map()
                messagebox.showinfo("Token Map Loaded", f"Token map loaded from {file_path}")
        except Exception as e:
            self.system_message(f"Load token map error: {e}")
            messagebox.showerror("Load Error", f"Failed to load token map:\n{e}")

    def display_token_map(self):
        if not hasattr(self, 'token_map_window') or not tk.Toplevel.winfo_exists(self.token_map_window):
            self.token_map_window = tk.Toplevel(self.root)
            self.token_map_window.title("Token Map Display")
            self.token_map_window.geometry("400x500")
            self.token_map_display = tk.Text(self.token_map_window, wrap=tk.WORD)
            self.token_map_display.pack(fill=tk.BOTH, expand=True)
            self.token_map_display.config(state='normal')
            self.token_map_display.insert(tk.END, "Token Mappings:\n\n")
            mapping = self.model.token_to_id
            for token, idx in sorted(mapping.items(), key=lambda x: x[1]):
                self.token_map_display.insert(tk.END, f"{token}: {idx}\n")
            self.token_map_display.config(state='disabled')
        else:
            self.token_map_window.lift()

    def load_hf_model(self):
        """Prompt the user for a HuggingFace model name or path and load it."""
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            messagebox.showerror("HF Model Error", "The transformers library is not installed.")
            return
        model_name = simpledialog.askstring("Load HuggingFace Model", "Enter HuggingFace model name or local path (e.g. gpt2, llama-2-7b):")
        if not model_name:
            return
        try:
            self.system_message(f"Loading HuggingFace model: {model_name}...")
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.hf_model = AutoModelForCausalLM.from_pretrained(model_name)
            device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
            self.hf_device = device
            if torch:
                self.hf_model.to(device)
            self.system_message(f"HuggingFace model '{model_name}' loaded on {device}.")
            self.model_type_var.set("HuggingFace")
        except Exception as e:
            self.hf_model = None
            self.hf_tokenizer = None
            self.hf_device = None
            messagebox.showerror("HF Model Load Error", f"Failed to load HuggingFace model:\n{e}")

    def load_local_model(self):
        """Load a local model file. GGUF/GGML models require llama_cpp; others will be loaded via transformers."""
        file_path = filedialog.askopenfilename(title="Select Local Model File", filetypes=[("Model Files", "*.gguf *.ggml *.bin *.pt"), ("All Files", "*.*")])
        if not file_path:
            return
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()
        if ext in [".gguf", ".ggml"]:
            if llama_cpp is not None:
                try:
                    self.system_message(f"Loading local GGUF/GGML model via llama_cpp: {filename}...")
                    self.local_model = llama_cpp.Llama(model_path=file_path, n_ctx=2048)
                    self.system_message(f"Local GGUF/GGML model '{filename}' loaded via llama_cpp.")
                    self.model_type_var.set("LocalGGUF")
                    return
                except Exception as e:
                    self.local_model = None
                    self.system_message(f"llama_cpp failed to load model: {e}\nTrying transformers fallback.")
            if AutoTokenizer is not None and AutoModelForCausalLM is not None:
                try:
                    model_dir = os.path.dirname(file_path) or "."
                    file_name_only = os.path.basename(file_path)
                    self.system_message(f"Attempting to load GGUF/GGML with transformers: {filename}...")
                    self.hf_tokenizer = AutoTokenizer.from_pretrained(model_dir, gguf_file=file_name_only)
                    self.hf_model = AutoModelForCausalLM.from_pretrained(model_dir, gguf_file=file_name_only)
                    device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
                    self.hf_device = device
                    if torch:
                        self.hf_model.to(device)
                    self.system_message(f"Local GGUF/GGML model '{filename}' loaded via transformers on {device}.")
                    self.model_type_var.set("HuggingFace")
                    return
                except Exception as e:
                    self.hf_model = None
                    self.hf_tokenizer = None
                    self.hf_device = None
                    messagebox.showerror("Local Model Load Error", f"Failed to load GGUF/GGML model via transformers:\n{e}")
                    return
            messagebox.showerror("Local Model Error", "Unable to load GGUF/GGML models: neither llama_cpp nor transformers with GGUF support are available.")
            return
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            messagebox.showerror("Local Model Error", "The transformers library is not installed; cannot load local HuggingFace models.")
            return
        try:
            model_dir = file_path
            if os.path.isfile(file_path):
                model_dir = os.path.dirname(file_path) or "."
            self.system_message(f"Loading local HuggingFace model from {model_dir}...")
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.hf_model = AutoModelForCausalLM.from_pretrained(model_dir)
            device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
            self.hf_device = device
            if torch:
                self.hf_model.to(device)
            self.system_message(f"Local HuggingFace model loaded from {model_dir} on {device}.")
            self.model_type_var.set("HuggingFace")
        except Exception as e:
            self.hf_model = None
            self.hf_tokenizer = None
            self.hf_device = None
            messagebox.showerror("Local Model Load Error", f"Failed to load local HuggingFace model:\n{e}")

    def run_hf_inference(self, prompt: str) -> str:
        """Generate a response with the loaded HuggingFace model. This is experiencing issues on build."""
        if not self.hf_model or not self.hf_tokenizer:
            raise RuntimeError("No HuggingFace model loaded.")
        if torch is None:
            raise RuntimeError("Torch is required for HuggingFace inference but is not available.")
        input_ids = self.hf_tokenizer.encode(prompt, return_tensors="pt").to(self.hf_device)
        max_new = max(1, self.max_tokens_var.get())
        temperature = max(1e-5, self.temperature_var.get())
        top_p = 0.95
        with torch.no_grad():
            output_ids = self.hf_model.generate(
                input_ids,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.hf_tokenizer.eos_token_id,
                eos_token_id=self.hf_tokenizer.eos_token_id
            )
        output_text = self.hf_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = output_text[len(prompt):].strip() if output_text.lower().startswith(prompt.lower()) else output_text.strip()
        return response if response else output_text.strip()

    def run_local_inference(self, prompt: str) -> str:
        if not self.local_model:
            raise RuntimeError("No local model loaded.")
        max_new = max(1, self.max_tokens_var.get())
        temperature = max(1e-5, self.temperature_var.get())
        top_p = 0.95
        try:
            if hasattr(self.local_model, "create_chat_completion"):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                result = self.local_model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_new,
                    temperature=temperature,
                    top_p=top_p
                )
                if isinstance(result, dict) and result.get("choices"):
                    msg_obj = result["choices"][0]
                    if isinstance(msg_obj, dict):
                        if "message" in msg_obj and isinstance(msg_obj["message"], dict):
                            text = msg_obj["message"].get("content", "")
                        else:
                            text = msg_obj.get("text", "")
                    else:
                        text = str(msg_obj)
                else:
                    text = ""
            else:
                result = self.local_model(
                    prompt,
                    max_tokens=max_new,
                    temperature=temperature,
                    top_p=top_p
                )
                if isinstance(result, dict) and result.get("choices"):
                    text = result["choices"][0].get("text", "")
                else:
                    text = str(result)
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"Local model inference error: {e}")

    def update_font_size(self):
        self.refresh_chat_display()

    def update_resource_usage(self):
        if self.process:
            cpu_usage_val = self.process.cpu_percent(interval=None)
            self.cpu_usage_label.config(text=f"CPU Usage: {cpu_usage_val:.1f}%")
        else:
            self.cpu_usage_label.config(text="CPU Usage: psutil not installed")
        self.gpu_usage_label.config(text="GPU Usage: N/A")
        self.root.after(1000, self.update_resource_usage)

def main():
    try:
        root = tk.Tk()
        app = QELMChatUI(root)
        root.mainloop() #Some builds seem to have issues with cmd, remove this if it gives you issues for a build.
    except Exception as e:
        err = f"Fatal error: {e}\n{traceback.format_exc()}"
        messagebox.showerror("Fatal Error", err)
        sys.exit(1)

if __name__ == "__main__": 
    main()
