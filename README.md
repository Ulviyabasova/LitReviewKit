# LitReviewKit v3

Run on Mac in the VS Code Terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run litreviewkit_app.py
```

Live search: Crossref, OpenAlex, Semantic Scholar. Import: Scopus and Web of Science CSV/XLSX exports with recognizable column names. The app screens term matches transparently, merges DOI/title duplicates, records per-query outcomes, offers manual screening, and exports Excel/CSV/protocol JSON. The optional literature chat uses an OpenAI API key supplied at runtime; requests send the question plus up to eight matching paper abstracts to the API. Without a key it returns matching papers. Do not upload confidential abstracts or restricted content without permission.

The app runs searches afresh. Session data disappears after server restart; export the workbook to keep it. Automatic decisions are suggestions, and incomplete abstracts can cause false exclusions. Citation counts can differ by source. Scopus/WoS live APIs, authenticated researcher identity, semantic embeddings, full-text analysis, journal rankings, and completed PRISMA flow are not implemented.
