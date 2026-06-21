# Customer Delivery Manifest

This directory is self-contained and can be published as its own GitHub repository. It does not require files from the parent Egyptian-university RAG project.

## Required deliverables

| File | Purpose |
|---|---|
| `schematic_diff_agent.py` | Complete ingestion, Chroma/FAISS/BM25 indexing, LangGraph QA, extraction, comparison, validation, and CSV implementation |
| `demo.py` | One-command end-to-end ingestion → indexing → question-answering demonstration |
| `test_schematic_diff_agent.py` | Deterministic automated test suite with external services mocked |
| `README.md` | Installation, configuration, architecture, and run instructions |
| `REPORT.md` | Part 0, Level 1, Level 2.1, design reasoning, limitations, and rubric checklist |
| `requirements.txt` | Complete Python dependency list |
| `.env.example` | Safe OpenAI and Chroma configuration template |
| `.gitignore` | Secrets, PDFs, caches, models, indexes, and generated outputs excluded |
| `data/README.md` | Customer PDF placement and data-security instructions |

## Pre-delivery verification

```powershell
python -m pip install -r requirements.txt
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest test_schematic_diff_agent.py -q -p no:cacheprovider
python -m ruff check schematic_diff_agent.py demo.py test_schematic_diff_agent.py
python -m py_compile schematic_diff_agent.py demo.py
python demo.py --help
```

## Customer actions before submission

- [ ] Copy `.env.example` to `.env` and set a real `OPENAI_API_KEY` locally.
- [ ] Supply approved OLD and NEW PDFs.
- [ ] Run `demo.py` and record exact document/page/chunk counts in `REPORT.md`.
- [ ] Review generated answers and impact findings with a hardware engineer.
- [ ] Initialize/push this folder as the delivery GitHub repository.
- [ ] Add the GitHub URL to the submission portal before the deadline.

Do not copy the parent repository's `school_rag`, Egyptian university dataset, notebook, indexes, or reports into this customer delivery; they are unrelated and may confuse evaluation.
