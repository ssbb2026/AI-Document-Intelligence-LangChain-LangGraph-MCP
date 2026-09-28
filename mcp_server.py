"""Read-only MCP tools; start separately with `python mcp_server.py`."""
from mcp.server.fastmcp import FastMCP
from rag_engine import search_documents, indexed_sources
from rag_graph import ask

mcp = FastMCP("Document Intelligence")

@mcp.tool()
def search_indexed_documents(query: str, top_k: int = 4) -> list[dict]:
    """Retrieve matching passages and their PDF source metadata."""
    return search_documents(query, top_k)

@mcp.tool()
def list_indexed_documents() -> list[str]:
    """List the PDFs ingested into the local vector index."""
    return indexed_sources()

@mcp.tool()
def answer_from_documents(question: str) -> dict:
    """Answer a question using the indexed PDFs, returning citations."""
    result = ask(question)
    return {"answer": result["answer"], "sources": result.get("sources", []),
            "verification": result.get("verification")}

if __name__ == "__main__":
    # Local stdio transport: configure an MCP client to launch this command.
    mcp.run(transport="stdio")
