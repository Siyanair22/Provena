"""
Query Router

Intelligently routes natural language queries to SQL, RAG, or HYBRID execution.

Two-phase classification:
1. Heuristic fast-path - pattern matching for obvious cases
2. LLM classification - for ambiguous queries

QueryOrchestrator handles end-to-end query processing with routing.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Literal


from provena.core import config
from provena.query import artifacts
from provena.query.decompose import decompose, verdict
from provena.query.sql_engine import FactCatalog, SQLGenerator, SafeSQLExecutor
from provena.query.retrieval import RetrievalError, Retriever
from provena.providers.llm import AnswerError, GroqLLM
from provena.memory import MemoryManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RouteType = Literal["SQL", "RAG", "HYBRID"]


class QueryRouter:
    """
    Classifies natural language queries to determine routing strategy.

    Uses two-phase approach:
    1. Heuristic pattern matching (fast)
    2. LLM classification (when ambiguous)
    """

    # SQL signal patterns (queries about structured data operations)
    SQL_SIGNALS = [
        r"\bhow many\b",
        r"\bcount\b",
        r"\btotal\b",
        r"\bsum\b",
        r"\baverage\b",
        r"\bmean\b",
        r"\bmedian\b",
        r"\blist all\b",
        r"\bshow all\b",
        r"\bcompare\b",
        r"\btable\b",
        r"\bcolumns?\b",
        r"\brows?\b",
        r"\bdata from\b",
        r"\bvalue[sd]?\b.*\bin\b",
        r"\bfilter\b",
        r"\bsort\b",
        r"\bgroup by\b",
        r"\bmax\b",
        r"\bmin\b",
    ]

    # RAG signal patterns (queries about understanding/explanation)
    RAG_SIGNALS = [
        r"\bexplain\b",
        r"\bdescribe\b",
        r"\bsummarize\b",
        r"\bwhat does\b",
        r"\bwhat is\b",
        r"\bwhy\b",
        r"\bhow does\b",
        r"\bwhat are the implications\b",
        r"\btell me about\b",
        r"\bmeaning of\b",
        r"\binterpret\b",
        r"\bdefine\b",
        r"\bconcept\b",
        r"\bmethodology\b",
        r"\bapproach\b",
        r"\bprocess\b",
    ]

    # Visualisation wording. Deliberately NOT added to SQL_SIGNALS: chart
    # choice is data-driven, but *reaching* the data is intent-driven, and
    # "plot revenue over time" matches none of the 20 SQL signals. These break
    # a one-weak-signal tie and never force a route on their own.
    VIZ_SIGNALS = [
        r"\bchart\b",
        r"\bplot\b",
        r"\bgraph\b",
        r"\bvisuali[sz]e\b",
        r"\btrend\b",
        r"\bover time\b",
        r"\bbreakdown\b",
        r"\bdistribution\b",
        r"\bby (month|quarter|year|region|category|segment|department)\b",
        r"\btop \d+\b",
    ]

    # Phrases that mean "a picture printed in the document", not "draw me one".
    # Without this veto, "explain the graph in Figure 3" would be sent to the
    # SQL engine and come back empty.
    VIZ_DOC_REFS = [
        r"\bfigure\s*\d",
        r"\b(figure|graph|chart|table)\s+(on|in)\s+(page|section)\b",
        r"\bthe (graph|chart|figure) (shows|shown|above|below)\b",
    ]

    def __init__(self):
        self.llm = GroqLLM()
        self.catalog = FactCatalog()

    def classify_heuristic(self, query: str) -> Optional[RouteType]:
        """
        Fast heuristic classification based on pattern matching.

        Args:
            query: Natural language query

        Returns:
            "SQL", "RAG", or None (if ambiguous)
        """
        query_lower = query.lower()

        # Count matches
        sql_matches = sum(1 for pattern in self.SQL_SIGNALS if re.search(pattern, query_lower))
        rag_matches = sum(1 for pattern in self.RAG_SIGNALS if re.search(pattern, query_lower))

        # Strong SQL signal
        if sql_matches >= 2 and rag_matches == 0:
            logger.info(f"Heuristic: SQL (matches={sql_matches})")
            return "SQL"

        # Strong RAG signal
        if rag_matches >= 2 and sql_matches == 0:
            logger.info(f"Heuristic: RAG (matches={rag_matches})")
            return "RAG"

        # Single strong SQL indicator
        if sql_matches == 1 and rag_matches == 0:
            strong_sql = [r"\bhow many\b", r"\bcount\b", r"\btotal\b", r"\bsum\b"]
            if any(re.search(p, query_lower) for p in strong_sql):
                logger.info("Heuristic: SQL (strong single match)")
                return "SQL"

        # Visualisation wording as a tie-breaker only. A viz word alone never
        # forces SQL, and a reference to a figure printed in the document vetoes.
        viz_matches = sum(1 for p in self.VIZ_SIGNALS if re.search(p, query_lower))
        refers_to_printed_figure = any(re.search(p, query_lower) for p in self.VIZ_DOC_REFS)

        if viz_matches and sql_matches >= 1 and rag_matches == 0 and not refers_to_printed_figure:
            logger.info(f"Heuristic: SQL (viz tie-break, viz={viz_matches}, sql={sql_matches})")
            return "SQL"

        # Ambiguous - fall back to LLM
        logger.info(f"Heuristic: AMBIGUOUS (sql={sql_matches}, rag={rag_matches})")
        return None

    def classify_llm(
        self,
        query: str,
        document_ids: Optional[List[str]] = None
    ) -> RouteType:
        """
        LLM-based classification when heuristics are ambiguous.

        Args:
            query: Natural language query
            document_ids: Optional document filter for context

        Returns:
            "SQL", "RAG", or "HYBRID"
        """
        # The tables most relevant to THIS question, not the first 1000
        # characters of the whole registry. The old truncation cut mid-table
        # once a few documents were loaded, so the model classified against
        # whichever tables happened to be listed first.
        table_context = self.catalog.render(self.catalog.candidates(query, document_ids))

        system_prompt = """You are a query classification expert for a hybrid RAG + SQL system.

Your task: Classify the user's query into ONE of these categories:
- SQL: Query asks for structured data operations (counts, totals, filtering, listing table data, aggregations)
- RAG: Query asks for explanations, summaries, or understanding of unstructured text content
- HYBRID: Query needs BOTH structured data AND conceptual understanding

Rules:
1. If the query explicitly references table names, columns, or asks for data values → SQL
2. If the query asks "why", "how does X work", "explain concept" → RAG
3. If the query asks "what does the data show about concept X" → HYBRID
4. Return JSON format: {"route": "SQL"|"RAG"|"HYBRID", "reasoning": "brief explanation"}

Available Tables:
__TABLE_CONTEXT__

Example classifications:
- "How many rows in table X?" → SQL
- "Show revenue by segment" → SQL
- "Explain the methodology" → RAG
- "What do the revenue figures tell us about market trends?" → HYBRID
"""

        user_message = f"""Query: {query}

Classify this query as SQL, RAG, or HYBRID. Return JSON only."""

        response = None
        try:
            response = self.llm.chat(
                messages=[
                    # A plain replace, not .format(). The prompt contains a
                    # literal JSON example - {"route": "SQL"|...} - and
                    # str.format reads those braces as a field, raising
                    # KeyError('"route"'). Classification then silently fell
                    # back to RAG for every ambiguous query, which is how a
                    # question like "show revenue by segment" reached the text
                    # retriever instead of the SQL engine.
                    {
                        "role": "system",
                        "content": system_prompt.replace(
                            "__TABLE_CONTEXT__", table_context
                        ),
                    },
                    {"role": "user", "content": user_message}
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=1500
            )

            # Parse JSON response
            result = json.loads(response)

            route = result.get("route", "RAG").upper()
            reasoning = result.get("reasoning", "")

            # Validate route
            if route not in ["SQL", "RAG", "HYBRID"]:
                logger.warning(f"Invalid route '{route}' from LLM, defaulting to RAG")
                route = "RAG"

            logger.info(f"LLM classification: {route} - {reasoning}")
            return route

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Response was: {response[:500] if response else 'None'}")
            return "RAG"
        except KeyError as e:
            logger.error(f"Missing key in JSON response: {e}")
            logger.error(f"Response was: {response[:500] if response else 'None'}")
            return "RAG"
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Response was: {response[:500] if response else 'None'}")
            return "RAG"

    def classify(
        self,
        query: str,
        document_ids: Optional[List[str]] = None
    ) -> RouteType:
        """
        Main classification method combining heuristic and LLM.

        Args:
            query: Natural language query
            document_ids: Optional document filter

        Returns:
            "SQL", "RAG", or "HYBRID"
        """
        # Try heuristic first
        heuristic_route = self.classify_heuristic(query)

        if heuristic_route is not None:
            return heuristic_route

        # Fall back to LLM for ambiguous cases
        return self.classify_llm(query, document_ids)




# How many top passages are checked for a table stub before a RAG answer is
# upgraded to HYBRID. Beyond the top few, a stub is incidental context.
BRIDGE_TOP_N = 3

MERGE_PROMPT = """You are answering a question from two sources: figures computed from the document's tables, and passages from its text.

Question: {query}

Computed from the tables:
{sql_answer}

From the document text:
{rag_answer}

Write one answer that presents the figures and uses the text to explain them.
Use only numbers that appear above; never compute or introduce new ones.
If the two sources disagree, say so. Be concise."""


class QueryOrchestrator:
    """
    End-to-end query processing with intelligent routing.

    Every response carries `segments`: the answer text split by where it came
    from, so the UI can mark what is verified.

      computed    prose written over SQL rows (the artifact beside it is exact)
      retrieved   prose written over cited passages
      unverified  a model merge of the two, with no single source
    """

    def __init__(self, memory: Optional[MemoryManager] = None):
        self.router = QueryRouter()
        self.sql_generator = SQLGenerator()
        self.sql_executor = SafeSQLExecutor()
        self.retriever = Retriever()
        self.llm = GroqLLM()
        self.memory = memory or MemoryManager()

    def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        force_route: Optional[RouteType] = None
    ) -> Dict[str, Any]:
        """
        Process a natural language query end-to-end.

        Args:
            query: Natural language question
            session_id: Session identifier for conversation history
            document_ids: Optional document filter
            force_route: Override classification (for testing)

        Returns:
            Dict with answer, route, segments, artifacts, sources, sql, etc.
        """
        logger.info(f"Processing query: {query}")

        original_query = query

        if session_id and self.memory.session_exists(session_id):
            history = self.memory.get_history(session_id)
            if history:
                logger.info("Rewriting follow-up query with conversation context")
                query = self.llm.rewrite_followup(query, history)
                logger.info(f"Rewritten query: {query}")

        route = force_route or self.router.classify(query, document_ids)
        logger.info(f"Route: {route}")

        if route == "SQL":
            result = self._execute_sql(query, document_ids)
        elif route == "RAG":
            result = self._execute_rag(query, document_ids)
        else:
            result = self._execute_hybrid(query, document_ids)

        # A failed answer is not remembered: a follow-up would otherwise be
        # rewritten against an error message.
        if session_id and not result.get("error"):
            self.memory.add_exchange(session_id, original_query, result.get("answer", ""))

        # Guaranteed on every route so a client never has to test for them.
        result.setdefault("artifacts", [])
        result.setdefault("segments", [])
        result["original_query"] = original_query
        if original_query != query:
            result["rewritten_query"] = query

        return result

    # ------------------------------------------------------------ SQL ------

    def _execute_sql(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        table_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute SQL-only query."""
        logger.info("Executing SQL route")

        sql_result = self.sql_generator.generate(query, document_ids, table_ids)

        if not sql_result.get("sql"):
            message = sql_result.get("explanation") or "Could not generate SQL for this question."
            if sql_result.get("error") == "No tables have been extracted yet.":
                message = "No tables have been extracted yet. Upload a document containing tables first."
            return {
                "answer": message,
                "route": "SQL",
                "error": message,
                "sql_pattern": sql_result.get("pattern"),
                "artifacts": [],
                "query": query,
            }

        pattern = sql_result.get("pattern")
        exec_result = self.sql_executor.execute(
            sql_result["sql"], sql_result.get("params"), max_rows=sql_result.get("max_rows")
        )

        # One self-repair attempt. Free-form SQL only: a template that failed
        # is a bug here, not something the model can fix.
        repaired = False
        if not exec_result["success"] and pattern == "custom":
            fixed = self.sql_generator.repair(
                sql_result["sql"], exec_result.get("error", ""), query
            )
            if fixed:
                logger.info("Retrying with repaired SQL")
                repaired = True
                exec_result = self.sql_executor.execute(fixed)
                if exec_result["success"]:
                    sql_result["sql"] = fixed

        self.sql_generator.record_execution(pattern, exec_result["success"], repaired=repaired)

        if not exec_result["success"]:
            message = f"The query could not be run: {exec_result.get('error')}"
            return {
                "answer": message,
                "route": "SQL",
                "error": message,
                "sql": sql_result["sql"],
                "sql_pattern": pattern,
                "artifacts": [],
                "query": query,
            }

        # Build the artifact BEFORE the prose, so both describe the same
        # sanitised values rather than the chart saying 4.53 and the text 4.5.
        arts = artifacts.build_artifact(
            exec_result,
            sql=exec_result.get("sql_executed"),
            tables_used=sql_result.get("tables_used", []),
            title=query,
        )

        # The rows are the answer; the prose is a summary of them. If the
        # summary fails the artifact still stands, so this is not an error.
        prose_error = None
        try:
            answer = self.sql_executor.format_results(exec_result, query, slots=sql_result.get("slots"))
        except AnswerError as exc:
            prose_error = str(exc)
            answer = (
                f"Found {exec_result['row_count']} result(s), shown below. "
                "A written summary could not be generated."
            )

        result = {
            "answer": answer,
            "route": "SQL",
            "sql": sql_result["sql"],
            "sql_executed": exec_result.get("sql_executed"),
            "sql_explanation": sql_result.get("explanation"),
            "sql_pattern": pattern,
            "tables_used": sql_result.get("tables_used", []),
            "row_count": exec_result["row_count"],
            "artifacts": arts,
            "segments": [{"kind": "computed", "text": answer}],
            "query": query,
        }
        if prose_error:
            result["prose_error"] = prose_error
        return result

    # ------------------------------------------------------------ RAG ------

    def _execute_rag(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the RAG route, upgrading to HYBRID when it lands on a table.

        excise_tables() leaves a stub carrying table_id where each table body
        was. When one of the top passages is such a stub the answer lives in
        that table, and the text route alone can only paraphrase the stub's two
        sample rows - so the table is queried too.
        """
        rag = self._rag_leg(query, document_ids)

        table_ids: List[str] = []
        for source in rag.get("_results", [])[:BRIDGE_TOP_N]:
            for table_id in (source.get("metadata") or {}).get("table_ids", []):
                if table_id not in table_ids:
                    table_ids.append(table_id)

        if table_ids and not rag.get("error"):
            logger.info(f"RAG hit table stub(s) {table_ids}; upgrading to HYBRID")
            sql = self._execute_sql(query, document_ids, table_ids=table_ids)
            if not sql.get("error"):
                merged = self._merge(query, sql, rag)
                merged["bridged_from"] = "RAG"
                return merged

        rag.pop("_results", None)
        return rag

    def _rag_leg(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Retrieve (per sub-question), answer, cite. Carries _results internally."""
        logger.info("Executing RAG route")

        sub_questions = decompose(self.llm, query)
        per_question_k = (
            config.RAG_TOP_K if len(sub_questions) == 1
            else max(2, config.RAG_TOP_K // len(sub_questions) + 1)
        )

        results: List[Dict[str, Any]] = []
        seen: Dict[str, int] = {}
        coverage = []
        try:
            for sub in sub_questions:
                found, reranked = self.retriever.retrieve(
                    sub, top_k=per_question_k, document_ids=document_ids
                )
                ids = []
                for record in found:
                    if record["id"] not in seen:
                        seen[record["id"]] = len(results)
                        results.append(record)
                    ids.append(seen[record["id"]] + 1)  # 1-based, matches [Source n]
                coverage.append({
                    "question": sub,
                    "verdict": verdict(found, reranked),
                    "source_ids": ids,
                })
        except RetrievalError as exc:
            message = str(exc)
            return {
                "answer": message,
                "route": "RAG",
                "error": message,
                "sources": [],
                "sub_questions": [],
                "artifacts": [],
                "query": query,
            }

        sources = [self._source(r) for r in results]
        base = {
            "route": "RAG",
            "sources": sources,
            "sub_questions": coverage if len(coverage) > 1 else [],
            "artifacts": [],
            "query": query,
            "_results": results,
        }

        if not results:
            answer = "I couldn't find any relevant information in the documents to answer this."
            return {**base, "answer": answer, "segments": [{"kind": "retrieved", "text": answer, "source_ids": []}]}

        context = self.retriever.build_context(results)
        if len(coverage) > 1:
            lines = [
                f"{i}. {c['question']} - retrieval: {c['verdict']}"
                + (f" (sources {', '.join(map(str, c['source_ids']))})" if c["source_ids"] else "")
                for i, c in enumerate(coverage, 1)
            ]
            context += (
                "\n\n---\n\nThe question has these parts:\n" + "\n".join(lines)
                + "\nAnswer each part. For a part marked NOT_FOUND, say the documents "
                "do not cover it. For PARTIAL, say the coverage is incomplete."
            )

        try:
            answer = self.llm.generate_answer(query, context)
        except AnswerError as exc:
            message = str(exc)
            return {**base, "answer": message, "error": message}

        return {
            **base,
            "answer": answer,
            "segments": [{
                "kind": "retrieved",
                "text": answer,
                "source_ids": list(range(1, len(sources) + 1)),
            }],
        }

    @staticmethod
    def _source(r: Dict[str, Any]) -> Dict[str, Any]:
        """A citation the UI can locate: file, page, section, offsets, snippet."""
        content = r.get("content") or ""
        return {
            "score": r.get("score"),
            "score_kind": r.get("score_kind"),
            "content": content[:300] + "..." if len(content) > 300 else content,
            "heading_context": r.get("heading_context", ""),
            "source_file": r.get("source_file", ""),
            "document_id": r.get("document_id"),
            "chunk_id": r.get("chunk_id"),
            "section": r.get("section"),
            "start_offset": r.get("start_offset"),
            "end_offset": r.get("end_offset"),
            "page_number": r.get("page_number"),
            "table_ids": (r.get("metadata") or {}).get("table_ids", []),
        }

    # --------------------------------------------------------- HYBRID ------

    def _execute_hybrid(
        self,
        query: str,
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute hybrid SQL + RAG query."""
        logger.info("Executing HYBRID route")
        sql = self._execute_sql(query, document_ids)
        rag = self._rag_leg(query, document_ids)
        return self._merge(query, sql, rag)

    def _merge(
        self,
        query: str,
        sql: Dict[str, Any],
        rag: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Combine the two legs.

        A failed leg is reported, never merged: its answer string is an error
        message, and feeding that to the merge prompt as if it were content is
        how HYBRID used to produce confident prose about "SQL execution failed".
        """
        rag.pop("_results", None)
        sql_ok, rag_ok = not sql.get("error"), not rag.get("error")
        rag_found = rag_ok and bool(rag.get("sources"))

        segments: List[Dict[str, Any]]
        if sql_ok and rag_found:
            try:
                answer = self.llm.chat(
                    messages=[{
                        "role": "user",
                        "content": MERGE_PROMPT.format(
                            query=query,
                            sql_answer=sql.get("answer", ""),
                            rag_answer=rag.get("answer", ""),
                        ),
                    }],
                    temperature=0.3,
                    max_tokens=1024,
                )
                segments = [{
                    "kind": "unverified",
                    "text": answer,
                    "note": "Written by the model from the computed figures and the cited passages.",
                }]
            except Exception as exc:  # noqa: BLE001 - fall back to both legs verbatim
                logger.error(f"Failed to merge results: {exc}")
                answer = (
                    f"**From the tables:**\n{sql['answer']}\n\n"
                    f"**From the text:**\n{rag['answer']}"
                )
                segments = sql.get("segments", []) + rag.get("segments", [])
        elif sql_ok:
            answer, segments = sql["answer"], sql.get("segments", [])
        elif rag_ok:
            answer, segments = rag["answer"], rag.get("segments", [])
        else:
            answer, segments = "Neither the tables nor the text could answer this question.", []

        # The artifact stays exact; the flag tells the UI the prose beside it
        # is a merge, several generative hops from the rows.
        arts = [dict(a) for a in sql.get("artifacts", [])]
        for art in arts:
            art["provenance"] = {
                **art["provenance"],
                "answer_is_llm_merged": sql_ok and rag_found,
                "source_documents": [
                    {"source_file": s.get("source_file", ""), "score": s.get("score")}
                    for s in rag.get("sources", [])
                ],
            }

        result = {
            "answer": answer,
            "route": "HYBRID",
            "sql": sql.get("sql"),
            "sql_executed": sql.get("sql_executed"),
            "sql_explanation": sql.get("sql_explanation"),
            "sql_pattern": sql.get("sql_pattern"),
            "sql_error": sql.get("error"),
            "rag_error": rag.get("error"),
            "tables_used": sql.get("tables_used", []),
            "row_count": sql.get("row_count"),
            "sources": rag.get("sources", []),
            "sub_questions": rag.get("sub_questions", []),
            "artifacts": arts,
            "segments": segments,
            "query": query,
        }
        if not sql_ok and not rag_ok:
            result["error"] = answer
        return result
