"""
swiftbasket_core.py
Reusable core business intelligence logic for SwiftBasket:
Hybrid SQL Analytics + Evidence-Gated RAG.
"""

import os
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

import pandas as pd
from sqlalchemy import create_engine, text
from fastembed import TextEmbedding
from google import genai

# =====================================================================
# 1. CONFIGURATION & CLIENT INITIALIZATION
# =====================================================================

# Load the database URL securely from the environment
DATABASE_URL = os.getenv("SWIFTBASKET_DATABASE_URL")
if not DATABASE_URL:
    raise ValueError(
        "Environment variable SWIFTBASKET_DATABASE_URL must be configured. "
        "Example: postgresql://postgres:YOUR_PASSWORD@localhost:5432/quick_commerce_bi"
    )

# The module uses a shared PostgreSQL engine so both SQL and RAG branches access the same database.
engine = create_engine(DATABASE_URL)

TABLE_NAME = "swiftbasket_embeddings"
EMBEDDING_DIMENSION = 384
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Initialize local embedding model for vector retrieval
embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)

# Initialize Google Gemini SDK client
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

# Relevance screening thresholds
SIMILARITY_ACCEPT_THRESHOLD = 0.65
SIMILARITY_REVIEW_THRESHOLD = 0.55

STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with", 
    "is", "are", "was", "were", "what", "which", "who", "when", "where", "how", 
    "tell", "me", "show", "give", "do", "you", "have", "any", "about", "this", 
    "that", "it", "from", "by", "i", "can", "please"
}


# =====================================================================
# 2. DYNAMIC RELATIONAL SCHEMA INSPECTION
# =====================================================================

def _load_relational_schema(db_engine) -> pd.DataFrame:
    """
    Dynamically discovers non-system relational tables (excluding vector tables).
    Inspects the actual schema (e.g., swiftbasket).
    """
    schema_query = text("""
        SELECT 
            c.table_schema,
            c.table_name, 
            c.column_name, 
            c.ordinal_position, 
            c.data_type, 
            c.is_nullable
        FROM information_schema.columns c
        JOIN information_schema.tables t 
          ON c.table_schema = t.table_schema AND c.table_name = t.table_name
        WHERE c.table_schema NOT IN ('information_schema', 'pg_catalog')
          AND c.table_schema NOT LIKE 'pg_toast%'
          AND c.table_name != :vector_table
          AND t.table_type = 'BASE TABLE'
        ORDER BY c.table_schema, c.table_name, c.ordinal_position;
    """)
    try:
        with db_engine.connect() as conn:
            return pd.read_sql(schema_query, conn, params={"vector_table": TABLE_NAME})
    except Exception:
        return pd.DataFrame()

# Inspected relational schema DataFrame cached at module load
schema_df = _load_relational_schema(engine)


# =====================================================================
# 3. DATA STRUCTURES & RETRIEVAL HELPERS
# =====================================================================

@dataclass
class RetrievedDocument:
    """Representation of a retrieved document matching notebook expectations."""
    id: Any
    text: str
    metadata: Dict[str, Any]
    similarity: float


def extract_content_tokens(text_str: Any) -> Set[str]:
    """Extract lowercase alphanumeric tokens excluding basic stop words."""
    if not text_str:
        return set()
    tokens = re.findall(r'\b[a-zA-Z0-9_]+\b', str(text_str).lower())
    return {t for t in tokens if t not in STOP_WORDS and len(t) > 1}


def compute_evidence_overlap(query_text: str, results: List[Any]) -> Tuple[int, Set[str], float]:
    """
    Computes lexical term overlap between query tokens and retrieved documents.
    """
    query_tokens = extract_content_tokens(query_text)
    if not query_tokens:
        return 0, set(), 0.0

    combined_doc_text = " ".join([
        str(getattr(r, "text", "")) + " " + str(getattr(r, "metadata", ""))
        for r in results
    ]).lower()
    
    doc_tokens = set(re.findall(r'\b[a-zA-Z0-9_]+\b', combined_doc_text))
    matched = query_tokens.intersection(doc_tokens)
    overlap_ratio = len(matched) / len(query_tokens)
    
    return len(query_tokens), matched, overlap_ratio


def _format_schema_context(df: pd.DataFrame) -> str:
    """Dynamically formats schema_df into a readable string for the LLM prompt."""
    table_blocks = []
    relational_df = df[df['table_name'] != TABLE_NAME]
    
    for (schema_name, table_name), group in relational_df.groupby(['table_schema', 'table_name'], sort=True):
        cols = [f"{row['column_name']} ({row['data_type']})" for _, row in group.iterrows()]
        table_blocks.append(f"Table: {schema_name}.{table_name}\nColumns: {', '.join(cols)}")
    return "\n\n".join(table_blocks)


def _is_transient_gemini_error(e: Exception) -> bool:
    """Identifies if an exception is a transient Gemini API error (e.g., HTTP 429 or 503)."""
    err_str = str(e).lower()
    return any(keyword in err_str for keyword in ["429", "503", "too many requests", "unavailable", "quota", "overloaded"])


# =====================================================================
# 4. CORE APPLICATION FUNCTIONS
# =====================================================================

def route_query(user_query: str) -> Dict[str, Any]:
    """
    Deterministic rule-based query router for SwiftBasket Hybrid SQL + RAG.
    Classifies queries into 'SQL' (analytical) or 'RAG' (specific records/context).
    """
    query_lower = user_query.lower()
    
    result = {
        "route": "RAG",
        "reason": "Ambiguous query lacking clear analytical signals; defaulting to conservative RAG.",
        "matched_signals": ["default"],
        "exact_identifier": None
    }
    
    # 1. Exact Order Identifier -> RAG (Rule: ORD followed by exactly 7 digits)
    exact_id_match = re.search(r'\bord\d{7}\b', query_lower)
    if exact_id_match:
        result["route"] = "RAG"
        result["reason"] = "Exact order identifier detected."
        result["matched_signals"] = ["ORD + 7 digits pattern"]
        result["exact_identifier"] = exact_id_match.group(0).upper()
        return result

    # 2. RAG Specific Phrases -> RAG
    rag_keywords = [
        "show me", "find an order", "details of", "what was purchased", 
        "status of", "payment details of", "delivery details of", 
        "find a product", "show a product"
    ]
    matched_rag = [kw for kw in rag_keywords if re.search(rf'\b{kw}\b', query_lower)]
    if matched_rag:
        result["route"] = "RAG"
        result["reason"] = "Query asks for a specific record or contextual evidence."
        result["matched_signals"] = matched_rag
        return result

    # 3. SQL Specific Phrases -> SQL
    sql_keywords = [
        "how many", "count", "total", "sum", "average", "mean", "median", 
        "maximum", "minimum", "highest", "lowest", "top", "ranking", "rate", 
        "percentage", "proportion", "trend", "growth", "compare", "comparison", 
        "by month", "by city", "by category", "by brand", "distribution",
        "which orders", "which products", "which customers"
    ]
    matched_sql = [kw for kw in sql_keywords if re.search(rf'\b{kw}\b', query_lower)]
    if matched_sql:
        result["route"] = "SQL"
        result["reason"] = "Query asks for analysis, aggregation, metrics, or comparisons across the dataset."
        result["matched_signals"] = matched_sql
        return result

    return result


def execute_sql_query(sql_query: str, max_rows: int = 100) -> Tuple[pd.DataFrame, str]:
    """
    Executes read-only analytical SQL queries against the SwiftBasket database.
    """
    if not isinstance(sql_query, str):
        return pd.DataFrame(), "Error: Query must be a string."
        
    q = sql_query.strip()
    if not q:
        return pd.DataFrame(), "Error: Empty query."
        
    # Strip comments and literal strings to safely inspect keywords
    q_clean = re.sub(r"'.*?'", "", q, flags=re.DOTALL)
    q_clean = re.sub(r"--.*?\n", "", q_clean)
    q_clean = re.sub(r"/\*.*?\*/", "", q_clean, flags=re.DOTALL)
    q_upper = q_clean.upper().strip()
    
    # 1. Enforce Read-Only Starting Keywords
    if not (q_upper.startswith("SELECT") or q_upper.startswith("WITH")):
        return pd.DataFrame(), "Error: Only SELECT or WITH analytical queries are allowed."
        
    # 2. Block DML/DDL Keywords
    banned_keywords = r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|MERGE|CALL|COPY)\b"
    if re.search(banned_keywords, q_upper):
        return pd.DataFrame(), "Error: Query contains forbidden DML/DDL keywords."
        
    # 3. Block Multi-statement Queries
    parts = [p.strip() for p in q_upper.split(";") if p.strip()]
    if len(parts) > 1:
        return pd.DataFrame(), "Error: Multiple SQL statements are not allowed in a single execution."
        
    # 4. Safe Execution and Fetching
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = result.fetchmany(max_rows)
            df = pd.DataFrame(rows, columns=result.keys())
        return df, "SUCCESS"
    except Exception as e:
        return pd.DataFrame(), f"Execution Error: {str(e).strip()}"


def generate_sql_query(user_query: str) -> str:
    """
    Converts a natural-language analytical question into a safe, read-only PostgreSQL query.
    Uses schema_df dynamically to ground Gemini in available tables and columns.
    """
    if not user_query or not user_query.strip():
        raise ValueError("User query cannot be empty.")

    schema_context = _format_schema_context(schema_df)

    prompt = f"""You are a PostgreSQL analytics SQL generator for the SwiftBasket database. Generate one read-only SQL query that answers the user's analytical question using only the supplied schema.

DATABASE SCHEMA:
The database schema name is 'swiftbasket'. All tables must be schema-qualified (e.g., swiftbasket.orders).
Do NOT use or reference 'swiftbasket_embeddings'.

{schema_context}

STRICT RULES:
1. Return PostgreSQL-compatible SQL ONLY.
2. The query MUST begin with SELECT or WITH.
3. No DDL/DML operations allowed (no INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, GRANT, REVOKE, MERGE, CALL, COPY).
4. Exactly one SQL statement. Do not chain multiple queries with semicolons.
5. Use ONLY the tables and columns explicitly listed in the schema above. Do NOT invent columns or tables.
6. Return raw SQL only. No markdown, no triple backticks (```), no explanations, no comments.
7. If the user's question cannot be answered using the provided schema, return exactly the word: UNANSWERABLE

USER QUESTION:
{user_query.strip()}

SQL QUERY:"""

    # Note: Exceptions from client.models.generate_content will natively propagate up.
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    
    raw_text = response.text.strip() if response and response.text else ""
    if not raw_text:
        raise ValueError("Gemini returned an empty response.")

    clean_sql = re.sub(r"^```(?:sql)?\s*", "", raw_text, flags=re.IGNORECASE)
    clean_sql = re.sub(r"\s*```$", "", clean_sql).strip()

    if clean_sql.upper() == "UNANSWERABLE":
        raise ValueError("The question cannot be answered with the available relational schema.")

    inspected_text = re.sub(r"'.*?'", "", clean_sql, flags=re.DOTALL)
    inspected_text = re.sub(r"--.*?\n", "", inspected_text)
    inspected_text = re.sub(r"/\*.*?\*/", "", inspected_text, flags=re.DOTALL).strip()
    upper_inspected = inspected_text.upper()

    if not (upper_inspected.startswith("SELECT") or upper_inspected.startswith("WITH")):
        raise ValueError("Generated SQL must begin with SELECT or WITH.")

    banned_keywords = r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|MERGE|CALL|COPY)\b"
    if re.search(banned_keywords, upper_inspected):
        raise ValueError("Generated SQL contains prohibited write/DDL keywords.")

    statements = [stmt.strip() for stmt in inspected_text.split(";") if stmt.strip()]
    if len(statements) > 1:
        raise ValueError("Multiple SQL statements detected in generated output.")

    if TABLE_NAME in clean_sql.lower():
        raise ValueError("Generated SQL attempted to access vector embeddings table.")

    return clean_sql


def hybrid_retrieve(query: str, top_k: int = 5) -> List[RetrievedDocument]:
    """
    Hybrid retrieval:
    1. Exact Order ID matches via PostgreSQL metadata lookup (similarity = 1.0)
    2. Semantic vector retrieval via BGE embeddings and pgvector cosine distance
    """
    # 1. Exact Identifier Check: ORD + 7 digits
    exact_match = re.search(r'\bord\d{7}\b', query.lower())
    if exact_match:
        order_id = exact_match.group(0).upper()
        lookup_sql = text(f"""
            SELECT id, text, metadata
            FROM {TABLE_NAME}
            WHERE metadata->>'order_id' = :order_id
            LIMIT 1;
        """)
        with engine.connect() as conn:
            row = conn.execute(lookup_sql, {"order_id": order_id}).fetchone()
            if row:
                doc_meta = row[2] if isinstance(row[2], dict) else json.loads(row[2])
                return [RetrievedDocument(id=row[0], text=row[1], metadata=doc_meta, similarity=1.0)]

    # 2. Semantic Vector Search
    query_vector = list(embedding_model.embed([query]))[0].tolist()
    vector_sql = text(f"""
        SELECT 
            id, 
            text, 
            metadata, 
            1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
        FROM {TABLE_NAME}
        ORDER BY embedding <=> CAST(:query_vec AS vector) ASC
        LIMIT :top_k;
    """)
    
    with engine.connect() as conn:
        rows = conn.execute(
            vector_sql, 
            {"query_vec": json.dumps(query_vector), "top_k": top_k}
        ).fetchall()
        
        results = []
        for r in rows:
            meta = r[2] if isinstance(r[2], dict) else json.loads(r[2])
            results.append(RetrievedDocument(
                id=r[0],
                text=r[1],
                metadata=meta,
                similarity=float(r[3])
            ))
        return results


def enhanced_relevance_aware_retrieve(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Enhanced retrieval wrapper combining:
    1. Exact Identifier Routing
    2. Vector Cosine Similarity
    3. Rule-Based Lexical Evidence Overlap
    """
    raw_results = hybrid_retrieve(query, top_k=top_k)
    
    output = {
        "query": query,
        "retrieval_method": "UNKNOWN",
        "results": [],
        "status": "ABSTAIN",
        "top_similarity": 0.0,
        "average_similarity": 0.0,
        "reason": "",
        "exact_identifier": None
    }
    
    # 0. Detect exact identifier for downstream propagation
    exact_match = re.search(r'\bord\d{7}\b', query.lower())
    if exact_match:
        output["exact_identifier"] = exact_match.group(0).upper()
    
    if not raw_results:
        output["retrieval_method"] = "EXACT_OR_EMPTY"
        output["status"] = "ABSTAIN"
        output["reason"] = "No matching records found in database."
        return output

    # 1. Exact Identifier Match Check
    first_score = float(getattr(raw_results[0], "similarity", 0.0))
    if first_score == 1.0:
        output["retrieval_method"] = "EXACT_IDENTIFIER"
        output["results"] = raw_results
        output["status"] = "ACCEPT"
        output["top_similarity"] = 1.0
        output["average_similarity"] = 1.0
        output["reason"] = "Exact identifier match verified in metadata."
        return output

    # 2. Semantic Vector Similarity Evaluation
    output["retrieval_method"] = "SEMANTIC_VECTOR"
    scores = [float(getattr(r, "similarity", 0.0)) for r in raw_results]
    top_sim = max(scores) if scores else 0.0
    avg_sim = (sum(scores) / len(scores)) if scores else 0.0
    
    output["top_similarity"] = round(top_sim, 4)
    output["average_similarity"] = round(avg_sim, 4)

    # 3. Rule-Based Evidence Grounding Check
    q_token_count, matched_tokens, overlap_ratio = compute_evidence_overlap(query, raw_results)

    if top_sim < SIMILARITY_REVIEW_THRESHOLD:
        output["status"] = "ABSTAIN"
        output["results"] = []
        output["reason"] = f"Top similarity ({top_sim:.4f}) is below review threshold ({SIMILARITY_REVIEW_THRESHOLD}). Context withheld."
        
    elif top_sim >= SIMILARITY_ACCEPT_THRESHOLD:
        if q_token_count > 0 and overlap_ratio == 0.0:
            output["status"] = "REVIEW"
            output["results"] = raw_results
            output["reason"] = f"High similarity ({top_sim:.4f}) but zero lexical evidence overlap. Flagged for downstream validation."
        else:
            output["status"] = "ACCEPT"
            output["results"] = raw_results
            output["reason"] = f"High similarity ({top_sim:.4f}) supported by lexical evidence overlap ({overlap_ratio:.1%})."
            
    else:
        # Borderline similarity (0.55 <= top_sim < 0.65)
        if q_token_count >= 2 and overlap_ratio == 0.0:
            output["status"] = "ABSTAIN"
            output["results"] = []
            output["reason"] = f"Borderline similarity ({top_sim:.4f}) with zero keyword/entity support in retrieved text. Context withheld."
        elif q_token_count <= 1 or overlap_ratio < 0.40:
            output["status"] = "REVIEW"
            output["results"] = raw_results
            output["reason"] = f"Borderline similarity ({top_sim:.4f}) with partial/ambiguous evidence ({overlap_ratio:.1%}). Requires downstream verification."
        else:
            output["status"] = "ACCEPT"
            output["results"] = raw_results
            output["reason"] = f"Borderline vector similarity ({top_sim:.4f}) compensated by strong lexical grounding ({overlap_ratio:.1%})."

    return output


def final_rag_answer(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    End-to-end grounded RAG pipeline.
    Integrates exact identifier routing, semantic search, evidence gating, and LLM generation.
    """
    retrieval_payload = enhanced_relevance_aware_retrieve(query, top_k=top_k)
    
    status = retrieval_payload.get("status", "ABSTAIN")
    method = retrieval_payload.get("retrieval_method", "UNKNOWN")
    results = retrieval_payload.get("results", [])
    top_sim = retrieval_payload.get("top_similarity", 0.0)
    avg_sim = retrieval_payload.get("average_similarity", 0.0)
    exact_id = retrieval_payload.get("exact_identifier")
    
    sources = [str(getattr(r, "id", "N/A")) for r in results]
    
    if status == "ABSTAIN":
        return {
            "query": query,
            "answer": "I don't have enough information in the retrieved data.",
            "retrieval_method": method,
            "retrieval_status": status,
            "top_similarity": top_sim,
            "average_similarity": avg_sim,
            "sources": [],
            "exact_identifier": exact_id
        }
        
    context_blocks = []
    for r in results:
        doc_id = str(getattr(r, "id", "N/A"))
        doc_text = str(getattr(r, "text", ""))
        doc_meta = str(getattr(r, "metadata", ""))
        context_blocks.append(f"--- Document ID: {doc_id} ---\nContent: {doc_text}\nMetadata: {doc_meta}")
        
    context_str = "\n\n".join(context_blocks)
    
    prompt = f"""You are a helpful and strictly factual AI assistant for SwiftBasket, an e-commerce platform.
Your task is to answer the user's question based strictly on the provided retrieved context.

RULES:
1. Answer ONLY from the retrieved context provided below.
2. Do not invent facts, numbers, customers, orders, products, dates, or conclusions.
3. If the context does not contain enough information to fully answer the question, respond EXACTLY with: "I don't have enough information in the retrieved data."
4. The retrieved context may be flagged for review. Be highly conservative before stating a fact is true.

RETRIEVED CONTEXT:
{context_str}

USER QUESTION:
{query}

ANSWER:"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        answer_text = response.text.strip() if response and response.text else "I don't have enough information in the retrieved data."
    except Exception as e:
        if _is_transient_gemini_error(e):
            raise  # Let the transient API error bubble up to hybrid_answer and caller
        answer_text = f"[API/SYSTEM ERROR] Failed to generate answer: {str(e)}"
        
    return {
        "query": query,
        "answer": answer_text,
        "retrieval_method": method,
        "retrieval_status": status,
        "top_similarity": top_sim,
        "average_similarity": avg_sim,
        "sources": sources,
        "exact_identifier": exact_id
    }


def hybrid_answer(user_query: str) -> Dict[str, Any]:
    """
    Orchestrates the SwiftBasket Enterprise Assistant by routing the user's question
    to either the SQL Analytical pipeline or the RAG Contextual pipeline.
    """
    try:
        routing_info = route_query(user_query)
        route = routing_info.get("route", "RAG")
        reason = routing_info.get("reason", "No reason provided")
        signals = routing_info.get("matched_signals", [])
        exact_id = routing_info.get("exact_identifier")
    except Exception as e:
        return {
            "query": user_query,
            "route": "UNKNOWN",
            "exact_identifier": None,
            "answer_type": "ERROR",
            "status": "ERROR",
            "error": f"Routing failed: {str(e)}"
        }

    # 1. SQL Branch
    if route == "SQL":
        try:
            generated_sql = generate_sql_query(user_query)
            df, exec_status = execute_sql_query(generated_sql, max_rows=100)
            is_success = (exec_status == "SUCCESS")
            
            return {
                "query": user_query,
                "route": route,
                "route_reason": reason,
                "matched_signals": signals,
                "exact_identifier": exact_id,
                "answer_type": "SQL_RESULT",
                "generated_sql": generated_sql,
                "data": df if is_success else pd.DataFrame(),
                "status": "SUCCESS" if is_success else "ERROR",
                "error": exec_status if not is_success else None
            }
        except Exception as e:
            if _is_transient_gemini_error(e):
                raise  # Propagate transient API errors out
            return {
                "query": user_query,
                "route": route,
                "route_reason": reason,
                "matched_signals": signals,
                "exact_identifier": exact_id,
                "answer_type": "SQL_RESULT",
                "generated_sql": None,
                "data": pd.DataFrame(),
                "status": "ERROR",
                "error": f"SQL Pipeline Error: {str(e)}"
            }

    # 2. RAG Branch
    else:
        try:
            rag_payload = final_rag_answer(user_query, top_k=5)
            return {
                "query": user_query,
                "route": route,
                "route_reason": reason,
                "matched_signals": signals,
                "exact_identifier": exact_id,
                "answer_type": "RAG_RESULT",
                "answer": rag_payload.get("answer"),
                "sources": rag_payload.get("sources", []),
                "retrieval_status": rag_payload.get("retrieval_status", "UNKNOWN"),
                "retrieval_method": rag_payload.get("retrieval_method", "UNKNOWN"),
                "top_similarity": rag_payload.get("top_similarity"),
                "average_similarity": rag_payload.get("average_similarity"),
                "status": "SUCCESS",
                "error": None
            }
        except Exception as e:
            if _is_transient_gemini_error(e):
                raise  # Propagate transient API errors out
            return {
                "query": user_query,
                "route": route,
                "route_reason": reason,
                "matched_signals": signals,
                "exact_identifier": exact_id,
                "answer_type": "RAG_RESULT",
                "answer": None,
                "sources": [],
                "retrieval_status": "ERROR",
                "retrieval_method": "ERROR",
                "top_similarity": None,
                "average_similarity": None,
                "status": "ERROR",
                "error": f"RAG Pipeline Error: {str(e)}"
            }