"""Gradio application for Hugging Face Spaces."""
import gradio as gr
from rag_engine import ingest_uploaded_pdf
from rag_graph import ask

def upload_pdf(file_path):
    if not file_path:
        return "Choose a PDF to upload."
    try:
        filename, count = ingest_uploaded_pdf(file_path)
        return f"Indexed **{filename}** ({count} chunks)."
    except Exception as exc:
        return f"Upload failed: {type(exc).__name__}: {exc}"

def respond(message, history):
    try:
        result = ask(message)
        answer = result["answer"]
        sources = result.get("sources", [])
        verification = result.get("verification")
        if verification:
            answer += "\n\n**Grounding check:** " + ("Passed" if verification["verified"] else "Unverified — answer withheld")
        if sources:
            answer += "\n\n### Retrieved sources\n"
            for i, source in enumerate(sources, start=1):
                answer += (
                    f"\n- [Source {i}] **{source['source']}**, "
                    f"chapter: {source['chapter'] or '—'}, "
                    f"section: {source['section'] or '—'}, "
                    f"chunk: {source['chunk_index']}, "
                    f"similarity: {source['similarity']:.3f}"
                )
        return answer
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"

with gr.Blocks(title="AI Document Intelligence") as demo:
    gr.Markdown("# AI Document Intelligence — LangChain + LangGraph + MCP")
    gr.Markdown("Upload a PDF, then ask questions grounded in its contents.")
    upload = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
    upload_status = gr.Markdown()
    upload.upload(upload_pdf, inputs=upload, outputs=upload_status)
    gr.ChatInterface(fn=respond, type="messages")

if __name__ == "__main__":
    demo.launch()
