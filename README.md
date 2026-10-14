# LLM Chess

Play chess against a large language model in a desktop window (Tkinter).

## Setup

```bash
pip install -r requirements.txt
python llm_chess.py
```

Tkinter ships with the standard Python installers on Windows and macOS (on Linux: `sudo apt install python3-tk`).

## How to play

1. Paste your API key. The provider is detected from the key prefix; you can change it in the **Provider** box.
2. Click **Load models** to fetch the models available to your key, then choose one.
3. Choose your color and click **New game**.
4. Click a piece, then a destination square. Legal targets are shown with dots.

The **LLM replies** panel shows the model's raw answers. If the model gives an illegal or unreadable move,
it is asked again (up to 3 tries); after that a random legal move is played and the log says so.

| Provider       | Key prefix                 |
|----------------|----------------------------|
| Anthropic      | `sk-ant-`                  |
| OpenAI         | `sk-`, `sk-proj-`          |
| Google Gemini  | `AIza`                     |
| xAI            | `xai-`                     |
| Groq           | `gsk_`                     |
| OpenRouter     | `sk-or-`                   |
| Mistral        | 32-char key (pick manually if needed) |
| DeepSeek       | `sk-` (select manually)    |

Your API key is kept in memory only and sent only to the selected provider.
