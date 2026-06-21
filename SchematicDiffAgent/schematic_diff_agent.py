#!/usr/bin/env python3
"""
schematic_diff_agent.py
=======================

Hybrid-RAG agent for comparing two versions of a hardware (domain-controller)
schematic and reporting what changed at the SoC <-> peripheral connection level,
so a software team knows what they must re-configure.

Design:
    * PERSISTENT CORPUS        -> Chroma stores chunks, metadata, and dense vectors
    * LOCAL retrieval brain   -> all-mpnet-base-v2 + FAISS cosine + BM25 via RRF
    * REASONING brain         -> OpenAI Responses API reads retrieved text and only
                                 the relevant rendered page images (multimodal)
    * CHAT ORCHESTRATION      -> typed LangGraph rewrite -> retrieve -> generate

Pipeline = separate CLI commands, run one at a time:
    1) ingest      (run once per version)   -> text chunks + page images
    2) build-index (run once, after ingest) -> FAISS (dense) + BM25 (sparse)
    3) extract     (run once per version)   -> structured connections.md via OpenAI
    4) compare     (run once)               -> impact_report.md via OpenAI
    5) query       (optional, ad-hoc)       -> raw hybrid-RAG lookup
    6) validate    (optional)               -> evidence audit of extracted ports
    7) chat        (interactive question)   -> grounded answer + sources

See README.md for install + the full command sequence.
"""

import os
import re
import csv
import base64
import glob
import hashlib
import json
import pickle
import logging
import argparse
from typing import NotRequired, TypedDict

import numpy as np

import fitz  # PyMuPDF
import faiss
import chromadb
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from sentence_transformers import SentenceTransformer
from langgraph.graph import END, START, StateGraph

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None


# --------------------------------------------------------------------------- #
#  Prompts handed to the OpenAI reasoning brain
# --------------------------------------------------------------------------- #

EXTRACT_PROMPT = """You are a hardware-schematic extraction engine for an automotive
software team. From the retrieved schematic context (text) and the attached
rendered schematic page images, extract EVERY SoC <-> peripheral connection you
can identify.

Output ONLY GitHub-flavoured Markdown. For each interface, emit a table where
each row is ONE physical connection, keyed by (interface_instance, signal_role).
Use exactly these columns:

| interface_instance | signal_role | soc_port | net_name | peripheral_component | peripheral_type | peripheral_pin | power_domain | notes |

Rules:
- interface_instance  e.g. FlexCAN0, LLCE_CAN3, QSPI_A, DDR_A, DSPI1, I2C_0
- signal_role         e.g. TXD, RXD, CS, CLK, DATA0, SCK, SIN, SOUT, SDA, SCL
- soc_port            the SoC pin / port label exactly as printed (e.g. PB_01, QSPI_A_CS0)
- peripheral_component the exact part number / instance (e.g. TJA1043ATK, MX25UW51245GXDQ00, U17)
- peripheral_pin      the destination pin label (e.g. TXD, SI/SIO0, RST_N)
- power_domain        supplying rail if shown (e.g. VPRE_3V3, 1.8V); else leave blank
- notes               muxing, pull-ups, switch dependency, level shifters, etc.
- Quote pin/port labels verbatim. If a value is not shown, leave the cell blank.
- Do NOT invent connections. Extract only what the context/images support.
"""

COMPARE_PROMPT = """You are a hardware-change impact analyst for an automotive SOFTWARE
team. You are given two structured connection extractions of the SAME board:
an OLD version and a NEW version. Join rows on the key (interface_instance,
signal_role) and report what changed and what the software team must do.

Output ONLY Markdown, in this structure:

## Summary
One short paragraph: scope and risk level of the changes.

## Pin / Port Changes (IOMUX impact)
Table: interface_instance | signal_role | OLD soc_port | NEW soc_port | software action
(Only rows where soc_port changed.)

## Peripheral / Transceiver Changes (driver impact)
Table: interface_instance | OLD peripheral_component | NEW peripheral_component | software action
(Only rows where the peripheral part/type changed.)

## Added / Removed Connections
Two tables: ADDED (present only in NEW) and REMOVED (present only in OLD).

## Power-Domain Changes
Table of any power_domain differences (init-sequence impact).

## Unchanged (sanity)
One line stating how many connections matched identically.

Be precise and conservative. If a field is blank in one version, say "unspecified",
do not assume it is unchanged.
"""

INTERFACE_SEEDS = [
    "FlexCAN transceiver TXD RXD connection pin",
    "LLCE_CAN transceiver TJA1043 TJA1153 TJA1463 connection",
    "QSPI NOR flash connection data pins",
    "LPDDR4 DDR subsystem connection CKE CLK DQ",
    "SD eMMC uSDHC MAX4886 connection",
    "GMAC PFE_MAC1 SJA1105Q TJA1102 ethernet connection",
    "PFE_MAC0 PFE_MAC2 SJA1110 KSZ9031 AR8035 ethernet connection",
    "LIN transceiver TJA1124 connection",
    "FlexRay transceiver TJA1081 connection",
    "LLCE_SPI DSPI SJA1110 SJA1105 connection SCK SIN SOUT",
    "I2C VR5510 TCA9539 connection SDA SCL address",
    "USB USB83340 ULPI connection",
    "UART FT232RQ LIN connection",
    "GPIO PWM port usage",
    "power supply rail VR5510 BUCK LDO domain",
    "reset tree RESET_B POR connection",
]

REWRITE_PROMPT = """You are a search-query rewriting component for automotive hardware
schematics. Rewrite the user's question into a clear, specific English query optimized
for hybrid semantic and keyword retrieval. Preserve exact signal names, net names,
component references, pin labels, interface instances, numbers, and revision labels.
Do not answer and do not add facts. Return only the rewritten query."""

CHAT_PROMPT = """You are a grounded automotive hardware-schematic assistant. Answer
only from the supplied source-tagged schematic context and attached retrieved page
images. Never use unsupported outside knowledge. Cite factual claims with
[VERSION: label, SOURCE: filename, PAGE: N]. If the evidence is insufficient, say
exactly that. Be concise and identify likely software impact (IOMUX, driver, power
sequence, or configuration) only when the evidence supports it."""

INSUFFICIENT_ANSWER = "The retrieved schematic sources are insufficient to answer this question."


class SchematicRAGState(TypedDict):
    """Typed state shared by the chatbot's LangGraph nodes."""

    original_query: str
    version: NotRequired[str | None]
    k: NotRequired[int]
    openai_model: NotRequired[str | None]
    with_images: NotRequired[bool]
    rewritten_query: NotRequired[str]
    retrieved_chunks: NotRequired[list["RetrievedSchematicChunk"]]
    answer: NotRequired[str]
    sources: NotRequired[list["SchematicSource"]]
    error: NotRequired[str]


class RetrievedSchematicChunk(TypedDict):
    """One traceable hybrid-retrieval result in graph state."""

    score: float
    score_type: str
    dense_similarity: float | None
    bm25_score: float
    text: str
    version: str
    source: str
    page: int


class SchematicSource(TypedDict):
    """Normalized source citation returned with a grounded answer."""

    version: str
    source: str
    page: int


# --------------------------------------------------------------------------- #
#  Agent
# --------------------------------------------------------------------------- #

class SchematicDiffAgent:
    """Hybrid RAG: Chroma persistence, FAISS+BM25 retrieval, OpenAI reasoning."""

    def __init__(self, workdir: str = "./work",
                 model_name: str = "all-mpnet-base-v2",
                 event_logger: logging.Logger = None):
        """
        @brief   Initialize local retrieval resources and compile the chatbot graph.
        @param   workdir      Runtime artifact directory.
        @param   model_name   SentenceTransformer model used only for retrieval.
        @param   event_logger Optional application logger.
        """
        self.workdir = workdir
        self.index_dir = os.path.join(workdir, "_index")
        os.makedirs(self.index_dir, exist_ok=True)
        self.log = event_logger or logging.getLogger("SchematicDiffAgent")
        load_dotenv()
        chroma_path = os.environ.get("CHROMA_PATH", os.path.join(workdir, "_chroma"))
        self.collection_name = os.environ.get("CHROMA_COLLECTION", "schematic_chunks")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Local embedding brain (same model family as your RagTool.py)
        self.model = SentenceTransformer(model_name)

        # populated by _load_index()
        self.index = None        # FAISS dense index (cosine / inner-product)
        self.chunks = None        # list[str]
        self.metadata = None      # list[dict] -> {version, source, page}
        self.bm25 = None          # sparse index
        self.graph = self._build_chat_graph()

    # ---- embedding helpers ------------------------------------------------ #
    # NOTE: To switch the dense backend to Ollama, replace ONLY this method
    #       (call your local Ollama embeddings endpoint and return an
    #        L2-normalized float32 ndarray of shape [N, dim]).
    def _embed(self, texts):
        vecs = self.model.encode(texts, show_progress_bar=False)
        vecs = np.asarray(vecs, dtype="float32")
        faiss.normalize_L2(vecs)          # normalize -> inner product == cosine
        return vecs

    @staticmethod
    def _tokenize(text):
        return re.findall(r"[A-Za-z0-9_]+", text.lower())

    # ---- STAGE 1: ingest -------------------------------------------------- #
    def _get_chunks_from_pdf(self, file_path, filename, version, chunk_size, overlap):
        """Sliding-window chunking (same approach as RagTool.py)."""
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and 0 <= overlap < chunk_size")
        chunks, metadata = [], []
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            text = page.get_text()
            if not text:
                continue
            start = 0
            while start < len(text):
                chunk = text[start:start + chunk_size]
                chunks.append(chunk)
                metadata.append({"version": version, "source": filename, "page": i + 1})
                start += (chunk_size - overlap)
        doc.close()
        return chunks, metadata

    def _render_pages(self, file_path, out_dir, dpi):
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        doc = fitz.open(file_path)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat)
            p = os.path.join(out_dir, f"page_{i + 1:03d}.png")
            pix.save(p)
            paths.append(p)
        doc.close()
        return paths

    def ingest(self, label, pdf_path, chunk_size=1500, overlap=300, dpi=150):
        """
        @brief   Stage 1 — extract chunks and render page images for one version.
        @param   label       Version label such as OLD or NEW.
        @param   pdf_path    Input schematic PDF path.
        @param   chunk_size  Sliding-window size in characters.
        @param   overlap     Sliding-window overlap in characters.
        @param   dpi         Render resolution for page PNGs.
        @return  Summary dictionary; writes chunks.json and page images.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(pdf_path)
        vdir = os.path.join(self.workdir, label)
        os.makedirs(vdir, exist_ok=True)
        filename = os.path.basename(pdf_path)

        chunks, metadata = self._get_chunks_from_pdf(
            pdf_path, filename, label, chunk_size, overlap)
        images = self._render_pages(pdf_path, os.path.join(vdir, "pages"), dpi)

        with open(os.path.join(vdir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump({"chunks": chunks, "metadata": metadata, "images": images,
                       "source": filename}, f, ensure_ascii=False, indent=2)

        self.log.info("[ingest:%s] %d chunks, %d page images -> %s",
                      label, len(chunks), len(images), vdir)
        return {"label": label, "chunks": len(chunks), "images": len(images), "directory": vdir}

    # ---- STAGE 2: build-index -------------------------------------------- #
    def _load_all_chunks(self):
        chunks, metadata = [], []
        for cj in sorted(glob.glob(os.path.join(self.workdir, "*", "chunks.json"))):
            with open(cj, encoding="utf-8") as f:
                data = json.load(f)
            chunks.extend(data["chunks"])
            metadata.extend(data["metadata"])
        return chunks, metadata

    def build_index(self):
        """
        @brief   Stage 2 — build and persist dense FAISS plus sparse BM25 indexes.
        @return  Summary dictionary containing chunk count and embedding dimension.
        """
        if BM25Okapi is None:
            raise ImportError("rank_bm25 is required. pip install rank-bm25")
        chunks, metadata = self._load_all_chunks()
        if not chunks:
            raise FileNotFoundError("No ingested chunks found. Run `ingest` first.")

        # Dense (FAISS, cosine via normalized inner product)
        embeddings = self._embed(chunks)
        self._sync_chroma(chunks, metadata, embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, os.path.join(self.index_dir, "dense.faiss"))

        # Sparse (BM25)
        bm25 = BM25Okapi([self._tokenize(c) for c in chunks])

        with open(os.path.join(self.index_dir, "store.pkl"), "wb") as f:
            pickle.dump({"chunks": chunks, "metadata": metadata, "bm25": bm25}, f)

        self.log.info("Hybrid index built over %d chunks (dense dim=%d, + BM25 sparse).",
                      len(chunks), embeddings.shape[1])
        return {"chunks": len(chunks), "dimension": int(embeddings.shape[1])}

    def _sync_chroma(self, chunks, metadata, embeddings):
        dimension = int(embeddings.shape[1])
        stored_dimension = (self.collection.metadata or {}).get("dimension")
        if stored_dimension is not None and int(stored_dimension) != dimension:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                self.collection_name,
                metadata={"hnsw:space": "cosine", "dimension": dimension},
            )
        existing = self.collection.get(include=[])["ids"]
        if existing:
            self.collection.delete(ids=existing)
        ids = [hashlib.sha256(
            f"{meta.get('version')}|{meta.get('source')}|{meta.get('page')}|{chunk}".encode("utf-8")
        ).hexdigest()[:24] for chunk, meta in zip(chunks, metadata)]
        for start in range(0, len(chunks), 128):
            stop = start + 128
            self.collection.add(
                ids=ids[start:stop],
                documents=chunks[start:stop],
                metadatas=metadata[start:stop],
                embeddings=embeddings[start:stop].tolist(),
            )
        self.collection.modify(metadata={"hnsw:space": "cosine", "dimension": dimension})
        self.log.info("Mirrored %d indexed chunks into Chroma persistence.", len(chunks))

    def _load_index(self):
        di = os.path.join(self.index_dir, "dense.faiss")
        si = os.path.join(self.index_dir, "store.pkl")
        if not (os.path.exists(di) and os.path.exists(si)):
            raise FileNotFoundError("Index missing. Run `build-index` first.")
        self.index = faiss.read_index(di)

        # Dimension guard (same safety check style as RagTool.py)
        model_dim = self.model.get_sentence_embedding_dimension()
        if self.index.d != model_dim:
            raise ValueError(
                f"DB dimension mismatch! Index={self.index.d}, Model={model_dim}. "
                "Delete the _index folder and re-run `build-index`.")

        with open(si, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.metadata = data["metadata"]
        self.bm25 = data["bm25"]

    # ---- hybrid retrieval (dense cosine + BM25, fused with RRF) ----------- #
    def hybrid_search(self, query, k=10, pool=30, version=None, rrf_k=60):
        """
        @brief   Retrieve chunks with dense/BM25 Reciprocal Rank Fusion.
        @param   query    Natural-language or signal-oriented search query.
        @param   k        Maximum returned results.
        @param   pool     Minimum candidate pool per retriever.
        @param   version  Optional exact version filter.
        @param   rrf_k    RRF rank constant.
        @return  List of (fused_score, chunk_text, metadata), best first.
        """
        if not query.strip() or k <= 0:
            return []
        if self.index is None:
            self._load_index()

        qv = self._embed([query])
        candidate_count = self.index.ntotal if version else min(max(pool, k), self.index.ntotal)
        dense_scores, dense_idx = self.index.search(qv, candidate_count)
        dense_rank = {int(idx): r for r, idx in enumerate(dense_idx[0]) if idx != -1}
        dense_similarity = {
            int(idx): float(score)
            for idx, score in zip(dense_idx[0], dense_scores[0]) if idx != -1
        }

        sparse_scores = self.bm25.get_scores(self._tokenize(query))
        sparse_limit = len(sparse_scores) if version else min(max(pool, k), len(sparse_scores))
        sparse_idx = np.argsort(sparse_scores)[::-1][:sparse_limit]
        sparse_rank = {int(idx): r for r, idx in enumerate(sparse_idx)}

        # Reciprocal Rank Fusion (no score-scale assumptions needed)
        fused = {}
        for idx, r in dense_rank.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + r)
        for idx, r in sparse_rank.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + r)

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked:
            meta = self.metadata[idx]
            if version and meta.get("version") != version:
                continue
            scored_meta = {
                **meta,
                "score_type": "rrf",
                "dense_similarity": dense_similarity.get(idx),
                "bm25_score": float(sparse_scores[idx]),
            }
            results.append((score, self.chunks[idx], scored_meta))
            if len(results) >= k:
                break
        return results

    @staticmethod
    def _format_context(results):
        blocks = []
        for _, chunk, meta in results:
            src = f"VERSION: {meta.get('version')}, SOURCE: {meta.get('source')}, PAGE: {meta.get('page')}"
            blocks.append(f"[{src}]\nCONTENT: {chunk}")
        return "\n\n----------------\n\n".join(blocks)

    # ---- STAGE 5: query (ad-hoc) ----------------------------------------- #
    def query(self, q, k=10, version=None):
        """
        @brief   Stage 5 — return raw source-tagged hybrid retrieval context.
        @param   q        Search query.
        @param   k        Maximum results.
        @param   version  Optional version filter.
        @return  Formatted source-tagged context string.
        """
        results = self.hybrid_search(q, k=k, version=version)
        return self._format_context(results)

    # ---- OpenAI reasoning brain ------------------------------------------ #
    def _run_openai(self, prompt, image_paths=None, model=None):
        """Send text and optional retrieved-page images to the OpenAI API."""
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required; copy .env.example to .env")
        content = [{"type": "input_text", "text": prompt}]
        for image_path in image_paths or []:
            try:
                with open(image_path, "rb") as file:
                    encoded = base64.b64encode(file.read()).decode("ascii")
            except OSError as exc:
                self.log.error("Could not read schematic page image %s: %s", image_path, exc)
                raise RuntimeError(f"Could not read schematic page image: {image_path}") from exc
            content.append({"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"})
        try:
            response = OpenAI(timeout=900).responses.create(
                model=model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
                input=[{"role": "user", "content": content}],
                temperature=0.0,
            )
        except (OpenAIError, OSError, RuntimeError, ValueError) as exc:
            self.log.error("OpenAI reasoning call failed: %s", exc)
            raise RuntimeError(f"OpenAI reasoning call failed: {exc}") from exc
        return response.output_text.strip()

    # ---- STAGE 3: extract ------------------------------------------------- #
    def _select_retrieved_images(self, label, results, max_images):
        pages = sorted({int(result[2]["page"]) for result in results
                        if result[2].get("version") == label and result[2].get("page")})
        selected_pages = pages[:max_images]
        chunks_path = os.path.join(self.workdir, label, "chunks.json")
        with open(chunks_path, encoding="utf-8") as file:
            all_images = json.load(file).get("images", [])
        selected = [all_images[page - 1] for page in selected_pages
                    if 0 < page <= len(all_images)]
        skipped = pages[max_images:]
        self.log.info("[%s] selected retrieved image pages=%s; skipped=%s",
                      label, selected_pages, skipped)
        return selected

    def _select_chat_images(self, results, max_images=20):
        images = []
        labels = sorted({result[2].get("version") for result in results
                         if result[2].get("version")})
        for label in labels:
            remaining = max_images - len(images)
            if remaining <= 0:
                break
            images.extend(self._select_retrieved_images(label, results, remaining))
        return images

    def extract(self, label, out_path, k_per_seed=4, model=None, with_images=True,
                max_images=20):
        """
        @brief   Stage 3 — retrieve evidence and extract structured connections.
        @param   label       Version to extract.
        @param   out_path    Destination Markdown file.
        @param   k_per_seed  Results collected per interface seed.
        @param   model       Optional OpenAI reasoning model.
        @param   with_images Include only images for retrieved pages.
        @param   max_images  Maximum retrieved page images passed to OpenAI.
        @return  Summary dictionary describing generated output and evidence.
        """
        if max_images <= 0:
            raise ValueError("max_images must be greater than zero")
        if self.index is None:
            self._load_index()

        seen, results = set(), []
        for seed in INTERFACE_SEEDS:
            for r in self.hybrid_search(seed, k=k_per_seed, version=label):
                chunk = r[1]
                if chunk not in seen:
                    seen.add(chunk)
                    results.append(r)
        context = self._format_context(results)

        images = None
        if with_images:
            images = self._select_retrieved_images(label, results, max_images)

        prompt = (f"{EXTRACT_PROMPT}\n\n=== RETRIEVED SCHEMATIC CONTEXT "
                  f"(version {label}) ===\n{context}\n")
        md = self._run_openai(prompt, image_paths=images, model=model)

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        self.log.info("Extracted connections for '%s' -> %s (%d chunks, %d images).",
                      label, out_path, len(results), len(images or []))
        return {"label": label, "output": out_path, "context_chunks": len(results),
                "images": len(images or [])}

    # ---- STAGE 4: compare ------------------------------------------------- #
    def compare(self, old_md, new_md, out_path, model=None, export_csv=None):
        """
        @brief   Stage 4 — compare structured OLD/NEW connections and report impact.
        @param   old_md     OLD connections Markdown path.
        @param   new_md     NEW connections Markdown path.
        @param   out_path   Impact-report Markdown destination.
        @param   model      Optional OpenAI reasoning model.
        @param   export_csv Optional normalized four-table CSV destination.
        @return  Summary dictionary with written output paths.
        """
        with open(old_md, encoding="utf-8") as f:
            old = f.read()
        with open(new_md, encoding="utf-8") as f:
            new = f.read()
        prompt = (f"{COMPARE_PROMPT}\n\n=== OLD VERSION CONNECTIONS ===\n{old}\n\n"
                  f"=== NEW VERSION CONNECTIONS ===\n{new}\n")
        report = self._run_openai(prompt, model=model)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        if export_csv:
            self._export_diff_csv(report, export_csv)
        self.log.info("Impact report -> %s", out_path)
        return {"output": out_path, "csv": export_csv}

    @staticmethod
    def _markdown_tables(markdown):
        sections, heading = {}, ""
        lines = markdown.splitlines()
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            if line.startswith("##"):
                heading = line.lstrip("#").strip().lower()
            if line.startswith("|") and index + 1 < len(lines) and re.match(
                    r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$", lines[index + 1]):
                headers = [cell.strip().lower() for cell in line.strip("|").split("|")]
                index += 2
                rows = []
                while index < len(lines) and lines[index].strip().startswith("|"):
                    values = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                    rows.append(dict(zip(headers, values)))
                    index += 1
                sections.setdefault(heading, []).append(rows)
                continue
            index += 1
        return sections

    def _export_diff_csv(self, report, out_path):
        tables = self._markdown_tables(report)
        normalized = []
        for heading, groups in tables.items():
            if "pin / port" in heading:
                change_type = "PIN_CHANGE"
            elif "peripheral / transceiver" in heading:
                change_type = "PERIPHERAL_CHANGE"
            elif "added" in heading:
                change_type = "ADDED"
            elif "removed" in heading:
                change_type = "REMOVED"
            else:
                continue
            for rows in groups:
                for row in rows:
                    old_value = (row.get("old soc_port") or
                                 row.get("old peripheral_component") or row.get("old value", ""))
                    new_value = (row.get("new soc_port") or
                                 row.get("new peripheral_component") or row.get("new value", ""))
                    normalized.append({
                        "change_type": change_type,
                        "interface_instance": row.get("interface_instance", ""),
                        "signal_role": row.get("signal_role", ""),
                        "old_value": old_value,
                        "new_value": new_value,
                        "software_action": row.get("software action", ""),
                    })
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["change_type", "interface_instance",
                                    "signal_role", "old_value", "new_value", "software_action"])
            writer.writeheader()
            writer.writerows(normalized)
        self.log.info("Exported %d normalized diff rows -> %s", len(normalized), out_path)

    @staticmethod
    def _connection_rows(markdown):
        rows = []
        for groups in SchematicDiffAgent._markdown_tables(markdown).values():
            for table in groups:
                for row in table:
                    if {"interface_instance", "signal_role", "soc_port"}.issubset(row):
                        rows.append(row)
        return rows

    def validate(self, label, connections_path=None, out_path=None):
        """
        @brief   Validate extracted connection rows against top-five retrieved text.
        @param   label            Version label used to filter retrieval.
        @param   connections_path Optional connections.md path for the version.
        @param   out_path         Optional validation report destination.
        @return  Summary dictionary with checked and suspicious row counts.
        """
        connections_path = connections_path or os.path.join(self.workdir, label, "connections.md")
        out_path = out_path or os.path.join(self.workdir, label, "validation_report.md")
        with open(connections_path, encoding="utf-8") as file:
            rows = self._connection_rows(file.read())
        suspicious = []
        for row in rows:
            port = row.get("soc_port", "").strip()
            if not port:
                continue
            query = " ".join([row.get("interface_instance", ""), row.get("signal_role", ""), port])
            evidence = self.hybrid_search(query, k=5, version=label)
            if not any(port.lower() in chunk.lower() for _, chunk, _ in evidence):
                suspicious.append(row)
        lines = [f"# Validation Report — {label}", "",
                 f"Checked rows: {len(rows)}", f"Suspicious rows: {len(suspicious)}", ""]
        if suspicious:
            lines.extend(["| interface_instance | signal_role | soc_port | reason |",
                          "|---|---|---|---|"])
            lines.extend(
                f"| {row.get('interface_instance', '')} | {row.get('signal_role', '')} | "
                f"{row.get('soc_port', '')} | Port absent from top-5 retrieved chunks |"
                for row in suspicious
            )
        else:
            lines.append("No suspicious populated SoC-port rows were found.")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")
        self.log.info("Validated %d rows for %s; suspicious=%d", len(rows), label, len(suspicious))
        return {"checked": len(rows), "suspicious": len(suspicious), "output": out_path}

    def _rewrite_node(self, state: SchematicRAGState):
        query = state["original_query"].strip()
        if not query:
            return {"rewritten_query": "", "error": "Question cannot be empty."}
        try:
            rewritten = self._run_openai(
                f"{REWRITE_PROMPT}\n\nUSER QUERY:\n{query}", model=state.get("openai_model")
            ).strip()
            return {"rewritten_query": rewritten or query}
        except RuntimeError as exc:
            self.log.warning("Query rewrite failed; using original query: %s", exc)
            return {"rewritten_query": query,
                    "error": f"Query rewrite failed; used original query: {exc}"}

    def _retrieve_node(self, state: SchematicRAGState):
        query = state.get("rewritten_query", "").strip()
        if not query:
            return {"retrieved_chunks": []}
        results = self.hybrid_search(
            query, k=state.get("k", 10), version=state.get("version")
        )
        return {"retrieved_chunks": [
            {"score": float(score), "text": chunk, **meta}
            for score, chunk, meta in results
        ]}

    def _generate_node(self, state: SchematicRAGState):
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            return {"answer": INSUFFICIENT_ANSWER, "sources": []}
        results = [(chunk["score"], chunk["text"], {
            "version": chunk.get("version"), "source": chunk.get("source"),
            "page": chunk.get("page")}) for chunk in chunks]
        context = self._format_context(results)
        sources = []
        seen_sources = set()
        for chunk in chunks:
            key = (chunk.get("version"), chunk.get("source"), chunk.get("page"))
            if key not in seen_sources:
                seen_sources.add(key)
                sources.append({"version": key[0], "source": key[1], "page": key[2]})
        images = None
        if state.get("with_images", True):
            images = self._select_chat_images(results, max_images=20)
        prompt = (
            f"{CHAT_PROMPT}\n\nORIGINAL QUESTION:\n{state['original_query']}\n\n"
            f"REWRITTEN RETRIEVAL QUERY:\n{state['rewritten_query']}\n\n"
            f"RETRIEVED SCHEMATIC CONTEXT:\n{context}"
        )
        answer = self._run_openai(
            prompt, image_paths=images, model=state.get("openai_model")
        ).strip()
        return {"answer": answer or INSUFFICIENT_ANSWER, "sources": sources}

    def _build_chat_graph(self):
        builder = StateGraph(SchematicRAGState)
        builder.add_node("rewrite", self._rewrite_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("generate", self._generate_node)
        builder.add_edge(START, "rewrite")
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        return builder.compile()

    def chat(self, question, version=None, k=10, model=None, with_images=True):
        """
        @brief   Run the typed LangGraph chatbot path: rewrite → retrieve → generate.
        @param   question    User's natural-language schematic question.
        @param   version     Optional OLD/NEW (or other label) retrieval filter.
        @param   k           Maximum hybrid retrieval results.
        @param   model       Optional OpenAI reasoning model.
        @param   with_images Attach images only for retrieved pages when version is set.
        @return  Final typed graph state with answer, evidence, and sources.
        """
        if k <= 0:
            raise ValueError("k must be greater than zero")
        return self.graph.invoke({
            "original_query": question,
            "version": version,
            "k": k,
            "openai_model": model,
            "with_images": with_images,
        })


# --------------------------------------------------------------------------- #
#  CLI  (each pipeline stage is a separate sub-command)
# --------------------------------------------------------------------------- #

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Schematic IRAG (Chroma persistence + FAISS/BM25 retrieval + OpenAI reasoning).")
    parser.add_argument("--workdir", default="./work", help="Working directory (default ./work).")
    parser.add_argument("--model", default="all-mpnet-base-v2",
                        help="Local sentence-transformers embedding model.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="Stage 1: extract chunks + render page images for one version.")
    p_ing.add_argument("--label", required=True, help="Version label, e.g. OLD or NEW.")
    p_ing.add_argument("--pdf", required=True, help="Path to the schematic PDF.")
    p_ing.add_argument("--chunk-size", type=int, default=1500)
    p_ing.add_argument("--overlap", type=int, default=300)
    p_ing.add_argument("--dpi", type=int, default=150)

    sub.add_parser("build-index", help="Stage 2: build hybrid FAISS(dense)+BM25(sparse) index.")

    p_ext = sub.add_parser("extract", help="Stage 3: OpenAI builds connections.md for one version.")
    p_ext.add_argument("--label", required=True)
    p_ext.add_argument("--out", required=True)
    p_ext.add_argument("--openai-model", default=None, help="e.g. gpt-4.1-mini")
    p_ext.add_argument("--no-images", action="store_true", help="Text-only (skip page images).")
    p_ext.add_argument("--max-images", type=int, default=20,
                       help="Maximum retrieved page images sent to OpenAI.")

    p_cmp = sub.add_parser("compare", help="Stage 4: OpenAI diffs versions -> impact report.")
    p_cmp.add_argument("--old", required=True, help="OLD connections.md")
    p_cmp.add_argument("--new", required=True, help="NEW connections.md")
    p_cmp.add_argument("--out", required=True)
    p_cmp.add_argument("--openai-model", default=None)
    p_cmp.add_argument("--export-csv", default=None,
                       help="Optional normalized CSV destination.")

    p_q = sub.add_parser("query", help="Stage 5 (optional): ad-hoc hybrid-RAG lookup.")
    p_q.add_argument("--q", required=True)
    p_q.add_argument("--k", type=int, default=10)
    p_q.add_argument("--version", default=None, help="Restrict to one version label.")

    p_val = sub.add_parser("validate", help="Validate extracted SoC ports against retrieved evidence.")
    p_val.add_argument("--label", required=True)
    p_val.add_argument("--connections", default=None)
    p_val.add_argument("--out", default=None)

    p_chat = sub.add_parser("chat", help="Grounded LangGraph chatbot over indexed schematics.")
    p_chat.add_argument("--q", required=True)
    p_chat.add_argument("--version", default=None)
    p_chat.add_argument("--k", type=int, default=10)
    p_chat.add_argument("--openai-model", default=None)
    p_chat.add_argument("--no-images", action="store_true")

    args = parser.parse_args()
    agent = SchematicDiffAgent(workdir=args.workdir, model_name=args.model)

    if args.command == "ingest":
        result = agent.ingest(args.label, args.pdf, args.chunk_size, args.overlap, args.dpi)
        print(f"Ingested '{result['label']}': {result['chunks']} chunks, "
              f"{result['images']} page images.")
    elif args.command == "build-index":
        result = agent.build_index()
        print(f"Hybrid index built over {result['chunks']} chunks "
              f"(dense dim={result['dimension']}, + BM25 sparse).")
    elif args.command == "extract":
        result = agent.extract(args.label, args.out, model=args.openai_model,
                               with_images=not args.no_images, max_images=args.max_images)
        print(f"Extracted '{result['label']}' -> {result['output']} "
              f"({result['context_chunks']} chunks, {result['images']} images).")
    elif args.command == "compare":
        result = agent.compare(args.old, args.new, args.out, model=args.openai_model,
                               export_csv=args.export_csv)
        print(f"Impact report -> {result['output']}")
        if result["csv"]:
            print(f"CSV export -> {result['csv']}")
    elif args.command == "query":
        print(agent.query(args.q, k=args.k, version=args.version))
    elif args.command == "validate":
        result = agent.validate(args.label, args.connections, args.out)
        print(f"Validated {result['checked']} rows; suspicious={result['suspicious']} "
              f"-> {result['output']}")
    elif args.command == "chat":
        result = agent.chat(args.q, version=args.version, k=args.k,
                            model=args.openai_model, with_images=not args.no_images)
        print(f"Original query: {result['original_query']}")
        print(f"Rewritten query: {result['rewritten_query']}")
        if result.get("error"):
            print(f"Note: {result['error']}")
        print("\nRetrieved chunks:")
        for rank, chunk in enumerate(result.get("retrieved_chunks", []), 1):
            cosine = chunk.get("dense_similarity")
            cosine_text = f"{cosine:.6f}" if cosine is not None else "n/a"
            print(f"{rank}. RRF={chunk['score']:.6f} | cosine_similarity={cosine_text} | "
                  f"BM25={chunk.get('bm25_score', 0.0):.6f} | {chunk.get('version')} | "
                  f"{chunk.get('source')} | page {chunk.get('page')}")
            print(f"   {chunk['text'].replace(chr(10), ' ')[:220]}")
        print(f"\nAnswer:\n{result['answer']}\n\nSources:")
        for source in result.get("sources", []):
            print(f"[{source['version']}, {source['source']}, page {source['page']}]")


if __name__ == "__main__":
    main()
