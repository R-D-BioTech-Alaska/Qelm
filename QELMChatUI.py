#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QELM Conversational UI - Rudi judi

This program provides a modern chat interface for interacting with a
Quantum-Enhanced Language Model (QELM).
  
- B

"""

import os
import sys
import json
import traceback
import datetime
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt', quiet=True)

# Dummy neural network fallback for demonstration - New implement asks before executing
def neural_network_inference(prompt):
    return "Neural fallback response: " + prompt[::-1]

def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-12 else vec

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# --- QELM Backend ---
class QuantumLanguageModel:
    def __init__(self):
        self.vocab_size = None
        self.embed_dim = None
        self.hidden_dim = None
        self.embeddings = None
        self.token_to_id = {}
        self.id_to_token = {}
        self.W_out = None
        self.W_proj = None

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
        # If the model file is produced by your QELM trainer, it should have a version key "4.0" (if not then this needs to be changed.
        if "version" in model_dict and model_dict["version"] == "4.0":
            self.vocab_size = model_dict["vocab_size"]
            self.embed_dim = model_dict["embed_dim"]
            self.hidden_dim = model_dict["hidden_dim"]
            self.embeddings = np.array(model_dict["embeddings"], dtype=np.float32)
            self.W_out = np.array(model_dict["W_out"], dtype=np.float32) if "W_out" in model_dict else np.random.randn(self.vocab_size, self.embed_dim).astype(np.float32)*0.01
            self.W_proj = np.array(model_dict["W_proj"], dtype=np.float32) if ("W_proj" in model_dict and model_dict["W_proj"] is not None) else None
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
            # For older format models
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
            self.W_out = np.array(model_dict["W_out"], dtype=np.float32) if "W_out" in model_dict else np.random.randn(self.vocab_size, self.embed_dim).astype(np.float32)*0.01
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
            # If token is placeholder, generate a friendly token.
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
            # Support three cases:
            # 1. File has keys "token_to_id" and optionally "id_to_token"
            # 2. File has a "vocabulary" key
            # 3. File is directly a mapping from tokens to ids.
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
                # If keys are placeholders, generate friendly mapping instead of raising error.
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

    def run_inference(self, prompt, max_length=20):
        tokens = word_tokenize(prompt.lower())
        if tokens:
            input_ids = [self.token_to_id.get(token, self.token_to_id.get("<UNK>", 0)) for token in tokens]
        else:
            input_ids = [0]
        response_tokens = []
        current_ids = input_ids.copy()
        for _ in range(max_length):
            logits = self.forward(current_ids, use_residual=True)
            probabilities = softmax(logits)
            sampled_id = int(np.random.choice(self.vocab_size, p=probabilities))
            sampled_token = self.id_to_token.get(sampled_id, "<UNK>")
            response_tokens.append(sampled_token)
            current_ids = [sampled_id]
            if sampled_token in [".", "!", "?"]:
                break
        return " ".join(response_tokens)

# --- Chat UI ---
    def __init__(self, root):
        self.root = root
        self.root.title("QELM Chat")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        self.current_theme = "light"
        self.model = QuantumLanguageModel()
        self.conversations = {}
        self.current_convo = "Default"
        self.conversations[self.current_convo] = []
        self.create_ui()

    def create_ui(self):
        # Top Panel: Model selection and theme toggle
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        ttk.Label(top_frame, text="Model:", font=("Helvetica", 12, "bold")).pack(side=tk.LEFT, padx=(0,5))
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(top_frame, textvariable=self.model_var, width=50)
        self.model_combo['values'] = []
        self.model_combo.set("Select or Load a QELM (.json/.qelm) Model...")
        self.model_combo.pack(side=tk.LEFT, padx=5)
        load_model_btn = ttk.Button(top_frame, text="Load Model", command=self.load_model)
        load_model_btn.pack(side=tk.LEFT, padx=5)
        theme_btn = ttk.Button(top_frame, text="Toggle Theme", command=self.toggle_theme)
        theme_btn.pack(side=tk.RIGHT, padx=5)
        
        # Left Panel: Conversation list
        left_frame = ttk.Frame(self.root, width=250)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)
        ttk.Label(left_frame, text="Conversations", font=("Helvetica", 14, "bold")).pack(pady=5)
        self.convo_list = tk.Listbox(left_frame, font=("Helvetica", 12))
        self.convo_list.pack(fill=tk.BOTH, expand=True)
        self.convo_list.bind("<<ListboxSelect>>", self.switch_conversation)
        new_conv_btn = ttk.Button(left_frame, text="New Conversation", command=self.new_conversation)
        new_conv_btn.pack(pady=5)
        
        # Right Panel: Chat display and message input
        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.chat_display = tk.Text(right_frame, wrap=tk.WORD, state=tk.DISABLED, bg="#ffffff", relief=tk.FLAT)
        self.chat_display.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.chat_display.tag_configure("user", justify="right", foreground="#1a73e8", font=("Helvetica", 12, "bold"))
        self.chat_display.tag_configure("qelm", justify="left", foreground="#34a853", font=("Helvetica", 12, "bold"))
        self.chat_display.tag_configure("system", justify="center", foreground="#ea4335", font=("Helvetica", 10, "italic"))
        scrollbar = ttk.Scrollbar(right_frame, command=self.chat_display.yview)
        self.chat_display['yscrollcommand'] = scrollbar.set
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
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

    def toggle_theme(self):
        if self.current_theme == "light":
            self.root.configure(bg="#2e2e2e")
            self.chat_display.configure(bg="#3c3f41", fg="#ffffff")
            self.current_theme = "dark"
        else:
            self.root.configure(bg="#f0f0f0")
            self.chat_display.configure(bg="#ffffff", fg="#000000")
            self.current_theme = "light"

    def new_conversation(self):
        name = f"Conversation {len(self.conversations) + 1}"
        self.conversations[name] = []
        self.current_convo = name
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
        self.add_message("User", user_text)
        self.entry_var.set("")
        try:
            response = self.model.run_inference(user_text)
        except Exception as e:
            err = f"QELM inference failed: {e}"
            self.system_message(err)
            use_nn = messagebox.askyesno("Inference Error", f"{err}\nUse neural network fallback?")
            if use_nn:
                response = neural_network_inference(user_text)
            else:
                response = "<Error: Response generation failed>"
        self.add_message("QELM", response)

    def add_message(self, sender, text):
        timestamp = datetime.datetime.now().strftime("%H:%M")
        msg = (sender, timestamp, text)
        if self.current_convo not in self.conversations:
            self.conversations[self.current_convo] = []
        self.conversations[self.current_convo].append(msg)
        self.refresh_chat_display()

    def refresh_chat_display(self):
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        convo = self.conversations.get(self.current_convo, [])
        for sender, timestamp, text in convo:
            if sender == "User":
                self.chat_display.insert(tk.END, f"{timestamp} ", "small_right")
                bubble = tk.Label(self.chat_display, text=text, bg="#e8f0fe", fg="#1a73e8",
                                    font=("Helvetica", 12, "bold"), wraplength=500, padx=10, pady=5, bd=1, relief=tk.RIDGE)
                self.chat_display.window_create(tk.END, window=bubble)
                self.chat_display.insert(tk.END, "\n\n")
            elif sender == "QELM":
                self.chat_display.insert(tk.END, f"{timestamp} ", "small_left")
                bubble = tk.Label(self.chat_display, text=text, bg="#e6f4ea", fg="#34a853",
                                    font=("Helvetica", 12, "bold"), wraplength=500, padx=10, pady=5, bd=1, relief=tk.RIDGE)
                self.chat_display.window_create(tk.END, window=bubble)
                self.chat_display.insert(tk.END, "\n\n")
            else:
                self.chat_display.insert(tk.END, f"{timestamp} SYSTEM: {text}\n\n", "system")
        self.chat_display.tag_configure("small_right", font=("Helvetica", 8), foreground="#555555", justify="right")
        self.chat_display.tag_configure("small_left", font=("Helvetica", 8), foreground="#555555", justify="left")
        self.chat_display.tag_configure("system", font=("Helvetica", 10, "italic"), foreground="#ea4335", justify="center")
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)

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
                    f.write(f"[{timestamp}] {sender}: {text}\n")
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
                # If keys look like placeholder tokens, replace them with friendly tokens.
                if all(isinstance(k, str) and k.startswith("<TOKEN_") and k.endswith(">") for k in token_map.keys()):
                    self.model.token_to_id = token_map
                    # Use the same friendly mapping as in the backend.
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
        mapping = self.model.token_to_id
        self.token_map_display.config(state='normal')
        self.token_map_display.delete('1.0', tk.END)
        self.token_map_display.insert(tk.END, "Token Mappings:\n\n")
        for token, idx in sorted(mapping.items(), key=lambda x: x[1]):
            self.token_map_display.insert(tk.END, f"{token}: {idx}\n")
        self.token_map_display.config(state='disabled')

    def update_resource_usage(self):
        self.root.after(1000, self.update_resource_usage)

    def update_time_label(self):
        self.root.after(1000, self.update_time_label)

    def process_log_queue(self):
        self.root.after(100, self.process_log_queue)

# --- Main Program ---
# --- Looping is required for an executable build ---
def main():
    try:
        root = tk.Tk()
        app = QELMChatUI(root)
        root.mainloop()
    except Exception as e:
        err = f"Fatal error: {e}\n{traceback.format_exc()}"
        messagebox.showerror("Fatal Error", err)
        sys.exit(1)

if __name__ == "__main__":
    main()
