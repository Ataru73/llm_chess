"""Play chess against an LLM.

A Tkinter GUI: paste an API key, the provider is detected from the key's
prefix (and can be overridden), pick one of that provider's models, then play.

Supported providers: Anthropic, OpenAI, Google Gemini, xAI, Groq, OpenRouter,
Mistral, DeepSeek. Only dependency outside the standard library: python-chess.
"""

import json
import random
import re
import threading
import tkinter as tk
import urllib.error
import urllib.request
from tkinter import font as tkfont, messagebox, ttk

import chess
import chess.pgn

# --------------------------------------------------------------------------- #
# LLM providers
# --------------------------------------------------------------------------- #

HTTP_TIMEOUT = 120


def http_json(url, headers=None, payload=None):
    """GET (payload is None) or POST JSON and return the decoded response."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "LLMChess/1.0")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            msg = json.loads(body).get("error", body)
            if isinstance(msg, dict):
                msg = msg.get("message", body)
        except ValueError:
            msg = body
        raise RuntimeError(f"HTTP {e.code}: {str(msg)[:400]}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}") from None


class Provider:
    name = "base"

    def __init__(self, api_key):
        self.api_key = api_key.strip()

    def list_models(self):
        raise NotImplementedError

    def complete(self, model, system, prompt):
        raise NotImplementedError


class AnthropicProvider(Provider):
    name = "Anthropic"
    base = "https://api.anthropic.com/v1"

    def _headers(self):
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}

    def list_models(self):
        models, after = [], None
        while True:
            url = f"{self.base}/models?limit=100" + (f"&after_id={after}" if after else "")
            data = http_json(url, self._headers())
            models += [m["id"] for m in data.get("data", [])]
            if not data.get("has_more"):
                return models
            after = data.get("last_id")

    def complete(self, model, system, prompt):
        data = http_json(f"{self.base}/messages", self._headers(), {
            "model": model,
            "max_tokens": 2048,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        })
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


class OpenAICompatibleProvider(Provider):
    """Any provider exposing OpenAI-style /models and /chat/completions."""
    name = "OpenAI"
    base = "https://api.openai.com/v1"

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def _keep_model(self, model_id):
        return True

    def list_models(self):
        data = http_json(f"{self.base}/models", self._headers())
        ids = [m["id"] for m in data.get("data", [])]
        return sorted((i for i in ids if self._keep_model(i)), reverse=True)

    def complete(self, model, system, prompt):
        data = http_json(f"{self.base}/chat/completions", self._headers(), {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        })
        content = data["choices"][0]["message"].get("content") or ""
        if isinstance(content, list):  # some providers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content


class OpenAIProvider(OpenAICompatibleProvider):
    name = "OpenAI"
    base = "https://api.openai.com/v1"

    def _keep_model(self, model_id):
        skip = ("embedding", "whisper", "tts", "dall-e", "image", "audio",
                "realtime", "moderation", "transcribe", "search", "davinci", "babbage")
        chat = model_id.startswith(("gpt", "o1", "o3", "o4", "o5", "chatgpt"))
        return chat and not any(s in model_id for s in skip)


class XAIProvider(OpenAICompatibleProvider):
    name = "xAI"
    base = "https://api.x.ai/v1"

    def _keep_model(self, model_id):
        return "image" not in model_id and "imagine" not in model_id


class GroqProvider(OpenAICompatibleProvider):
    name = "Groq"
    base = "https://api.groq.com/openai/v1"

    def _keep_model(self, model_id):
        return not any(s in model_id for s in ("whisper", "tts", "guard", "playai"))


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "OpenRouter"
    base = "https://openrouter.ai/api/v1"

    def list_models(self):
        data = http_json(f"{self.base}/models", self._headers())
        return sorted(m["id"] for m in data.get("data", []))


class MistralProvider(OpenAICompatibleProvider):
    name = "Mistral"
    base = "https://api.mistral.ai/v1"

    def _keep_model(self, model_id):
        return not any(s in model_id for s in ("embed", "moderation", "ocr"))


class DeepSeekProvider(OpenAICompatibleProvider):
    name = "DeepSeek"
    base = "https://api.deepseek.com/v1"


class GeminiProvider(Provider):
    name = "Google Gemini"
    base = "https://generativelanguage.googleapis.com/v1beta"

    def _headers(self):
        return {"x-goog-api-key": self.api_key}

    def list_models(self):
        models, token = [], None
        while True:
            url = f"{self.base}/models?pageSize=1000" + (f"&pageToken={token}" if token else "")
            data = http_json(url, self._headers())
            for m in data.get("models", []):
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    models.append(m["name"].removeprefix("models/"))
            token = data.get("nextPageToken")
            if not token:
                return sorted(models, reverse=True)

    def complete(self, model, system, prompt):
        data = http_json(f"{self.base}/models/{model}:generateContent", self._headers(), {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        })
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"No candidates returned: {json.dumps(data)[:300]}")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))


PROVIDERS = {cls.name: cls for cls in (
    AnthropicProvider, OpenAIProvider, GeminiProvider, XAIProvider,
    GroqProvider, OpenRouterProvider, MistralProvider, DeepSeekProvider,
)}


def detect_provider(api_key):
    """Guess the provider from the key prefix. Returns a provider name or None."""
    k = api_key.strip()
    if k.startswith("sk-ant-"):
        return AnthropicProvider.name
    if k.startswith("sk-or-"):
        return OpenRouterProvider.name
    if k.startswith("AIza"):
        return GeminiProvider.name
    if k.startswith("xai-"):
        return XAIProvider.name
    if k.startswith("gsk_"):
        return GroqProvider.name
    if k.startswith(("sk-proj-", "sk-svcacct-", "sk-admin-")):
        return OpenAIProvider.name
    if k.startswith("sk-"):
        return OpenAIProvider.name  # DeepSeek keys also look like this; user can override
    if re.fullmatch(r"[A-Za-z0-9]{32}", k):
        return MistralProvider.name
    return None


# --------------------------------------------------------------------------- #
# Asking the LLM for a move
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are a strong chess engine playing a game against a human. "
    "When asked for a move, think as much as you need, then end your reply with a "
    "final line of the exact form:\nMOVE: <move in SAN>\n"
    "The move must be one of the legal moves provided."
)


def build_prompt(board, llm_color, last_error=None):
    game = chess.pgn.Game.from_board(board)
    moves_pgn = str(game.mainline_moves()) or "(no moves yet)"
    legal = ", ".join(board.san(m) for m in board.legal_moves)
    side = "White" if llm_color == chess.WHITE else "Black"
    prompt = (
        f"You are playing {side}. It is your move.\n\n"
        f"Game so far (PGN): {moves_pgn}\n"
        f"Current position (FEN): {board.fen()}\n\n"
        f"Board (uppercase = White, lowercase = Black, rank 8 at top):\n{board}\n\n"
        f"Your legal moves: {legal}\n\n"
        "Choose the best move. Finish with the line: MOVE: <your move in SAN>"
    )
    if last_error:
        prompt += f"\n\nNote: your previous answer was rejected ({last_error}). Pick one of the legal moves listed above."
    return prompt


def _try_parse(board, token):
    token = token.strip().strip(".,;:!?*\"'`()[]")
    token = re.sub(r"^\d+\.+", "", token)  # "12.Nf3" / "12...Nf3"
    token = token.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    if not token:
        return None
    for parse in (board.parse_san, board.parse_uci):
        try:
            return parse(token)
        except ValueError:
            continue
    return None


def parse_move(board, text):
    """Extract a legal move from the LLM reply, or None."""
    text = text.replace("**", "")
    # 1) Preferred: the explicit "MOVE: xxx" line (last occurrence wins).
    for match in reversed(re.findall(r"MOVE\s*[:\-]\s*([^\s]+)", text, re.IGNORECASE)):
        move = _try_parse(board, match)
        if move:
            return move
    # 2) Fallback: any token that is a legal move, scanning from the end.
    for token in reversed(re.split(r"\s+", text)):
        move = _try_parse(board, token)
        if move:
            return move
    return None


def ask_llm_for_move(provider, model, board, llm_color, log, attempts=3):
    """Returns (move, was_random). Retries on illegal/unparseable replies."""
    error = None
    for attempt in range(1, attempts + 1):
        reply = provider.complete(model, SYSTEM_PROMPT, build_prompt(board, llm_color, error))
        log(f"[{model}] attempt {attempt}:\n{reply.strip()}\n")
        move = parse_move(board, reply)
        if move:
            return move, False
        error = "no legal move could be read from it"
    return random.choice(list(board.legal_moves)), True


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #

SQUARE = 72
LIGHT, DARK = "#f0d9b5", "#b58863"
HIGHLIGHT_LAST = "#cdd26a"
HIGHLIGHT_SEL = "#829769"
CHECK_COLOR = "#e06666"
GLYPHS = {"k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟"}

# Font preferences per role; the first family installed on this system wins.
UI_FONTS = ("Segoe UI", "SF Pro Text", "Helvetica Neue", "DejaVu Sans", "Noto Sans", "Liberation Sans")
MONO_FONTS = ("Consolas", "Menlo", "DejaVu Sans Mono", "Noto Sans Mono", "Liberation Mono")
PIECE_FONTS = ("Segoe UI Symbol", "Apple Symbols", "DejaVu Sans", "Noto Sans Symbols 2", "FreeSerif")


def pick_font(candidates, fallback):
    """Return the first installed font family from candidates, else the family
    behind the named Tk font `fallback` (needs a Tk root)."""
    installed = set(tkfont.families())
    return next((f for f in candidates if f in installed),
                tkfont.nametofont(fallback).actual("family"))


class ChessApp:
    def __init__(self, root):
        self.root = root
        root.title("LLM Chess")
        root.minsize(940, 640)

        self.board = chess.Board()
        self.provider = None
        self.human_color = chess.WHITE
        self.flipped = False
        self.selected = None
        self.thinking = False
        self.board_locked = False  # set after resigning
        self.game_id = 0  # invalidates in-flight LLM replies after New game / Undo

        self.ui_font = pick_font(UI_FONTS, "TkDefaultFont")
        self.mono_font = pick_font(MONO_FONTS, "TkFixedFont")
        self.piece_font = pick_font(PIECE_FONTS, "TkDefaultFont")

        self._build_controls()
        self._build_board()
        self._build_side_panel()
        self.redraw()
        self.set_status("Enter an API key, load models, then press New game.")

    # ---- layout -------------------------------------------------------------

    def _build_controls(self):
        frm = ttk.LabelFrame(self.root, text="LLM setup", padding=8)
        frm.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 4))

        ttk.Label(frm, text="API key:").grid(row=0, column=0, sticky="w")
        self.key_var = tk.StringVar()
        self.key_var.trace_add("write", lambda *_: self._on_key_change())
        self.key_entry = ttk.Entry(frm, textvariable=self.key_var, show="•", width=52)
        self.key_entry.grid(row=0, column=1, sticky="we", padx=4)
        self.show_key = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Show", variable=self.show_key,
                        command=lambda: self.key_entry.config(show="" if self.show_key.get() else "•")
                        ).grid(row=0, column=2, padx=4)

        ttk.Label(frm, text="Provider:").grid(row=0, column=3, sticky="w", padx=(12, 0))
        self.provider_var = tk.StringVar()
        self.provider_combo = ttk.Combobox(frm, textvariable=self.provider_var, state="readonly",
                                           values=list(PROVIDERS), width=16)
        self.provider_combo.grid(row=0, column=4, padx=4)
        self.provider_combo.bind("<<ComboboxSelected>>", lambda _e: self._clear_models())
        self.load_btn = ttk.Button(frm, text="Load models", command=self.load_models)
        self.load_btn.grid(row=0, column=5, padx=4)

        ttk.Label(frm, text="Model:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(frm, textvariable=self.model_var, state="readonly", width=50)
        self.model_combo.grid(row=1, column=1, sticky="we", padx=4, pady=(6, 0))

        ttk.Label(frm, text="You play:").grid(row=1, column=3, sticky="w", padx=(12, 0), pady=(6, 0))
        self.color_var = tk.StringVar(value="White")
        ttk.Combobox(frm, textvariable=self.color_var, state="readonly",
                     values=["White", "Black", "Random"], width=16
                     ).grid(row=1, column=4, padx=4, pady=(6, 0))
        ttk.Button(frm, text="New game", command=self.new_game).grid(row=1, column=5, padx=4, pady=(6, 0))
        frm.columnconfigure(1, weight=1)

    def _build_board(self):
        body = ttk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=4)
        self.body = body
        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, anchor="n")
        self.canvas = tk.Canvas(left, width=SQUARE * 8 + 20, height=SQUARE * 8 + 20,
                                highlightthickness=0, bg="#2b2b2b")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.status_var = tk.StringVar()
        ttk.Label(left, textvariable=self.status_var, font=(self.ui_font, 11, "bold"),
                  wraplength=SQUARE * 8).pack(pady=6, anchor="w")

    def _build_side_panel(self):
        right = ttk.Frame(self.body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))

        btns = ttk.Frame(right)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Flip board", command=self.flip).pack(side=tk.LEFT)
        ttk.Button(btns, text="Undo move", command=self.undo).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Resign", command=self.resign).pack(side=tk.LEFT)
        ttk.Button(btns, text="Copy PGN", command=self.copy_pgn).pack(side=tk.LEFT, padx=4)

        ttk.Label(right, text="Moves").pack(anchor="w", pady=(8, 0))
        self.moves_text = tk.Text(right, height=10, width=40, wrap="word", state="disabled",
                                  font=(self.mono_font, 10))
        self.moves_text.pack(fill=tk.X)

        ttk.Label(right, text="LLM replies").pack(anchor="w", pady=(8, 0))
        log_frame = ttk.Frame(right)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, wrap="word", state="disabled", font=(self.mono_font, 9))
        sb = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # ---- provider / models --------------------------------------------------

    def _on_key_change(self):
        detected = detect_provider(self.key_var.get())
        if detected and detected != self.provider_var.get():
            self.provider_var.set(detected)
            self._clear_models()

    def _clear_models(self):
        self.model_combo.config(values=[])
        self.model_var.set("")
        self.provider = None

    def load_models(self):
        key, name = self.key_var.get().strip(), self.provider_var.get()
        if not key:
            messagebox.showwarning("API key", "Please enter an API key.")
            return
        if not name:
            messagebox.showwarning("Provider", "Could not detect the provider; please pick one.")
            return
        provider = PROVIDERS[name](key)
        self.load_btn.config(state="disabled")
        self.set_status(f"Loading {name} models...")

        def work():
            try:
                models = provider.list_models()
                self.root.after(0, lambda: self._models_loaded(provider, models))
            except Exception as e:  # noqa: BLE001 - surface any failure to the user
                err = str(e)
                self.root.after(0, lambda: self._models_failed(name, err))

        threading.Thread(target=work, daemon=True).start()

    def _models_loaded(self, provider, models):
        self.load_btn.config(state="normal")
        if not models:
            self._models_failed(provider.name, "the provider returned no chat models")
            return
        self.provider = provider
        self.model_combo.config(values=models)
        self.model_var.set(models[0])
        self.set_status(f"Loaded {len(models)} {provider.name} models. Pick one and press New game.")

    def _models_failed(self, name, err):
        self.load_btn.config(state="normal")
        self.set_status("Could not load models.")
        messagebox.showerror("Error", f"Could not load {name} models:\n{err}")

    # ---- game flow ----------------------------------------------------------

    def new_game(self):
        if not self.provider or not self.model_var.get():
            messagebox.showwarning("Model", "Load the models and select one first.")
            return
        choice = self.color_var.get()
        if choice == "Random":
            choice = random.choice(["White", "Black"])
        self.human_color = chess.WHITE if choice == "White" else chess.BLACK
        self.flipped = self.human_color == chess.BLACK
        self.board = chess.Board()
        self.selected = None
        self.thinking = False
        self.game_id += 1
        self.board_locked = False
        self._clear_log()
        self.log(f"New game: you play {choice}, LLM is {self.model_var.get()} ({self.provider.name}).\n")
        self.after_move()

    def after_move(self):
        self.selected = None
        self.redraw()
        self.update_moves()
        if self.board.is_game_over(claim_draw=True):
            self.show_result()
        elif self.board.turn == self.human_color:
            self.set_status("Your move.")
        else:
            self.request_llm_move()

    def request_llm_move(self):
        self.thinking = True
        game_id = self.game_id
        model = self.model_var.get()
        provider = self.provider
        board = self.board.copy()
        llm_color = not self.human_color
        self.set_status(f"{model} is thinking...")

        def log(msg):
            self.root.after(0, lambda: self.game_id == game_id and self.log(msg))

        def work():
            try:
                move, was_random = ask_llm_for_move(provider, model, board, llm_color, log)
                self.root.after(0, lambda: self._llm_moved(game_id, move, was_random))
            except Exception as e:  # noqa: BLE001
                err = str(e)
                self.root.after(0, lambda: self._llm_failed(game_id, err))

        threading.Thread(target=work, daemon=True).start()

    def _llm_moved(self, game_id, move, was_random):
        if game_id != self.game_id:
            return
        self.thinking = False
        san = self.board.san(move)
        if was_random:
            self.log(f"Could not get a legal move from the LLM; playing random move {san}.\n")
        self.board.push(move)
        self.log(f"LLM plays {san}\n" + "-" * 40 + "\n")
        self.after_move()

    def _llm_failed(self, game_id, err):
        if game_id != self.game_id:
            return
        self.thinking = False
        self.log(f"ERROR: {err}\n")
        if messagebox.askretrycancel("LLM error", f"The LLM request failed:\n{err}\n\nRetry?"):
            self.request_llm_move()
        else:
            self.set_status("LLM request failed. Press Undo or New game.")

    def show_result(self):
        outcome = self.board.outcome(claim_draw=True)
        if outcome.winner is None:
            text = f"Draw ({outcome.termination.name.replace('_', ' ').lower()})."
        elif outcome.winner == self.human_color:
            text = "Checkmate - you win!"
        else:
            text = "Checkmate - the LLM wins."
        self.set_status(f"Game over: {text}  Result {outcome.result()}")
        self.log(f"Game over: {text}\n")

    def undo(self):
        if not self.board.move_stack or self.board_locked:
            return
        # Cancel any pending LLM reply, then take back to the human's turn.
        self.game_id += 1
        self.thinking = False
        self.board.pop()
        if self.board.turn != self.human_color and self.board.move_stack:
            self.board.pop()
        if self.board.turn != self.human_color:  # LLM opens as White: let it move again
            self.after_move()
            return
        self.selected = None
        self.redraw()
        self.update_moves()
        self.set_status("Move taken back. Your move.")

    def resign(self):
        if self.provider is None or self.board.is_game_over():
            return
        self.game_id += 1
        self.thinking = False
        self.set_status("You resigned. Press New game to play again.")
        self.log("You resigned.\n")
        self.board_locked = True

    def copy_pgn(self):
        game = chess.pgn.Game.from_board(self.board)
        game.headers["White"] = "Human" if self.human_color == chess.WHITE else self.model_var.get()
        game.headers["Black"] = "Human" if self.human_color == chess.BLACK else self.model_var.get()
        self.root.clipboard_clear()
        self.root.clipboard_append(str(game))
        self.set_status("PGN copied to clipboard.")

    def flip(self):
        self.flipped = not self.flipped
        self.redraw()

    # ---- input --------------------------------------------------------------

    def square_at(self, x, y):
        col, row = (x - 10) // SQUARE, (y - 10) // SQUARE
        if not (0 <= col < 8 and 0 <= row < 8):
            return None
        file, rank = (7 - col, row) if self.flipped else (col, 7 - row)
        return chess.square(file, rank)

    def on_click(self, event):
        if (self.provider is None or self.thinking or self.board_locked
                or self.board.is_game_over() or self.board.turn != self.human_color):
            return
        sq = self.square_at(event.x, event.y)
        if sq is None:
            return
        piece = self.board.piece_at(sq)
        if self.selected is None:
            if piece and piece.color == self.human_color:
                self.selected = sq
        elif sq == self.selected:
            self.selected = None
        elif piece and piece.color == self.human_color:
            self.selected = sq
        else:
            move = chess.Move(self.selected, sq)
            if (self.board.piece_type_at(self.selected) == chess.PAWN
                    and chess.square_rank(sq) in (0, 7)):
                promo = self.ask_promotion()
                if promo is None:
                    return
                move.promotion = promo
            if move in self.board.legal_moves:
                self.log(f"You play {self.board.san(move)}\n")
                self.board.push(move)
                self.after_move()
                return
            self.selected = None
        self.redraw()

    def ask_promotion(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Promote to")
        dlg.transient(self.root)
        dlg.resizable(False, False)
        result = {"piece": None}
        for ptype, sym in ((chess.QUEEN, "q"), (chess.ROOK, "r"), (chess.BISHOP, "b"), (chess.KNIGHT, "n")):
            def choose(p=ptype):
                result["piece"] = p
                dlg.destroy()
            tk.Button(dlg, text=GLYPHS[sym], font=(self.piece_font, 32), width=2,
                      command=choose).pack(side=tk.LEFT, padx=4, pady=4)
        dlg.grab_set()
        self.root.wait_window(dlg)
        return result["piece"]

    # ---- drawing ------------------------------------------------------------

    def redraw(self):
        c = self.canvas
        c.delete("all")
        last = self.board.peek() if self.board.move_stack else None
        check_sq = self.board.king(self.board.turn) if self.board.is_check() else None
        targets = set()
        if self.selected is not None:
            targets = {m.to_square for m in self.board.legal_moves if m.from_square == self.selected}

        for row in range(8):
            for col in range(8):
                file, rank = (7 - col, row) if self.flipped else (col, 7 - row)
                sq = chess.square(file, rank)
                x0, y0 = 10 + col * SQUARE, 10 + row * SQUARE
                color = LIGHT if (file + rank) % 2 else DARK
                if last and sq in (last.from_square, last.to_square):
                    color = HIGHLIGHT_LAST
                if sq == self.selected:
                    color = HIGHLIGHT_SEL
                if sq == check_sq:
                    color = CHECK_COLOR
                c.create_rectangle(x0, y0, x0 + SQUARE, y0 + SQUARE, fill=color, outline="")

                piece = self.board.piece_at(sq)
                cx, cy = x0 + SQUARE / 2, y0 + SQUARE / 2
                if piece:
                    glyph = GLYPHS[piece.symbol().lower()]
                    font = (self.piece_font, int(SQUARE * 0.62))
                    if piece.color == chess.WHITE:
                        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
                            c.create_text(cx + dx * 1.5, cy + dy * 1.5, text=glyph, font=font, fill="#222")
                        c.create_text(cx, cy, text=glyph, font=font, fill="#fafafa")
                    else:
                        c.create_text(cx, cy, text=glyph, font=font, fill="#111")
                if sq in targets:
                    r = SQUARE * 0.14 if not piece else SQUARE * 0.44
                    if piece:
                        c.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#3b5b2b", width=4)
                    else:
                        c.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#4f6f3f", outline="")

        # Coordinates
        for i in range(8):
            file_label = chess.FILE_NAMES[7 - i if self.flipped else i]
            rank_label = str(i + 1 if self.flipped else 8 - i)
            c.create_text(10 + i * SQUARE + SQUARE - 6, 10 + 8 * SQUARE - 8, text=file_label,
                          font=(self.ui_font, 9, "bold"), fill="#444")
            c.create_text(16, 10 + i * SQUARE + 10, text=rank_label, font=(self.ui_font, 9, "bold"), fill="#444")

    def update_moves(self):
        game = chess.pgn.Game.from_board(self.board)
        text = str(game.mainline_moves())
        self.moves_text.config(state="normal")
        self.moves_text.delete("1.0", tk.END)
        self.moves_text.insert(tk.END, text)
        self.moves_text.see(tk.END)
        self.moves_text.config(state="disabled")

    def set_status(self, text):
        self.status_var.set(text)

    def log(self, text):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state="disabled")


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    ChessApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
