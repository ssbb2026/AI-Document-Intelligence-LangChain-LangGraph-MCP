---
title: Intelligence
emoji: 🔥
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: 6.28.0
python_version: '3.13'
app_file: app.py
pinned: false
short_description: AI DOC
---

# Document Intelligence — LangChain, LangGraph, MCP, CI/CD

## Local setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Upload a PDF using Gradio. On first use, the embedding model downloads; on first
answer, Qwen downloads. CPU generation is slow. A persistent FAISS index is stored
in `data/faiss_index` and is reused by both the app and the MCP server **when they
run on the same filesystem**. For production, use persistent storage; the default
Hugging Face Space filesystem can be ephemeral.

## MCP

```bash
python mcp_server.py
```

Example client configuration (adjust to your Python and working directory):

```json
{
  "mcpServers": {
    "document-intelligence": {
      "command": "python",
      "args": ["/ABSOLUTE/PATH/TO/mcp_server.py"]
    }
  }
}
```

The MCP server uses stdio. A remote Hugging Face Gradio Space does not by itself
expose an MCP endpoint; deploy and secure an HTTP MCP server separately if needed.

## Tests

```bash
pip install -r requirements-dev.txt
ruff check --select E4,E7,E9,F .
python -m pytest -q
```

## Retrieval evaluation

After indexing a PDF, copy `evaluation_data.example.json` to `evaluation_data.json`.
Replace the example chunk IDs with manually verified labels and run:

```bash
python evaluation.py
```

Chunk IDs are namespaced by the stored PDF filename, e.g. `abcd1234_earth.pdf:chunk_7`.
The small Qwen model can still hallucinate; similarity thresholds and prompts are
not a rigorous factual-grounding guarantee. Review answers against cited passages.

## CI/CD

Create GitHub repository secret `HF_TOKEN` with write permission to the Space
`ssbb2026/AI`, then push to `main`. The workflow tests and deploys the repository.
Do not commit PDFs containing private information, secrets, downloaded models, or
untrusted FAISS pickle files. If the Hugging Face Space requires a different Gradio
SDK version, align the README header and requirements.


## Hallucination detector

After Qwen generates an answer, LangGraph runs `verify` before returning it. Each answer sentence is checked for entailment against its cited retrieved passage (or all retrieved passages if uncited) using `facebook/bart-large-mnli`. An unsupported claim causes the entire answer to be withheld. Set `NLI_MODEL` and `ENTAILMENT_THRESHOLD` in environment variables. The NLI model downloads on first use and needs substantial memory; test on your deployment hardware. NLI scores are model estimates, not calibrated proof of factual correctness. The detector cannot detect errors already present in the PDF or prove the PDF itself is reliable.
