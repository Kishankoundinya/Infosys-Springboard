from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from backend.app.rag.vector_store import search_similar_chunks
from backend.app.utils.config import settings
from backend.app.utils.logging import logger

def retrieve_relevant_chunks(
    query: str,
    company: Optional[str] = None,
    fiscal_year: Optional[str] = None,
    document_id: Optional[str] = None,
    top_k: Optional[int] = None
) -> List[Document]:
    """
    Retrieve grounded chunks with strict metadata filters.
    """
    k = top_k or settings.TOP_K
    filters = {}
    if document_id:
        filters["document_id"] = document_id
    elif company and fiscal_year:
        filters = {"$and": [{"company": company}, {"year": fiscal_year}]}
    elif company:
        filters["company"] = company
    elif fiscal_year:
        filters["year"] = fiscal_year

    filter_dict = filters if filters else None
    logger.info(f"Retrieving chunks for query: '{query}' with filter: {filter_dict}, k={k}")
    
    docs = search_similar_chunks(query, top_k=k, filter_dict=filter_dict)
    
    # If filtered search returned nothing and filters were specific, try company-only or open search
    if not docs and filter_dict and "document_id" not in filter_dict:
        logger.info("Specific filter yielded 0 chunks. Retrying with broader company filter...")
        if company:
            docs = search_similar_chunks(query, top_k=k, filter_dict={"company": company})
        if not docs:
            docs = search_similar_chunks(query, top_k=k)

    logger.info(f"Retrieved {len(docs)} chunks.")
    return docs

def format_context_with_citations(documents: List[Document]) -> str:
    """
    Format retrieved documents into a context block with explicit source tags:
    [Source: document_name, p. page]
    """
    if not documents:
        return "No relevant context found in uploaded documents."

    formatted_pieces = []
    for i, doc in enumerate(documents, start=1):
        doc_name = doc.metadata.get("document_name", "Unknown Document")
        page = doc.metadata.get("page", "Unknown")
        company = doc.metadata.get("company", "")
        content = doc.page_content.strip()

        piece = (
            f"--- Document Snippet {i} ---\n"
            f"Source: {doc_name}, Page: {page}, Company: {company}\n"
            f"Content:\n{content}\n"
        )
        formatted_pieces.append(piece)

    return "\n".join(formatted_pieces)
