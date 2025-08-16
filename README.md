# Digikala-Reviews-RAG

A full, end-to-end **Retrieval-Augmented Generation (RAG)** pipeline for Persian e-commerce reviews (Digikala).
This repo cleans and summarizes product reviews, builds retrieval indexes (TF-IDF and embedding-based with **GLOT500**), auto-generates QA pairs in Persian, fine-tunes a dual-encoder for better retrieval, and evaluates everything with automatic and human metrics.

> خلاصه فارسی پایین هر بخش آمده است.

---

## What you get

* **Data cleaning & stats** for `digikala-products.csv` and `digikala-comments.csv` (dropping products with 0 comments, fixing noisy pros/cons, weak-summary filtering).&#x20;
* **Product-level summaries** (extractive TF-IDF over comments + templated metadata) saved to CSV/XLSX.
* **Retrievers**

  * **TF-IDF** baseline (+ evidence sentence extraction).
  * **GLOT500 (zero-shot)** sentence embeddings with sliding-window doc encoding + optional PCA whitening.
  * **GLOT500 (fine-tuned)** dual-encoder trained on auto-generated Persian QA pairs.
* **Question generation** pipelines (Persian QG models, LLM prompts, NER masking, EN QG option).
* **Evaluation**

  * Automatic: Hit\@K, MRR, per-question evidence, per-model comparisons.
  * Human: Label Studio ranking protocol; **TF-IDF** ≈ best mean rank, **fine-tuned GLOT500** improves over zero-shot.&#x20;
* **Duplicate-QA detection** over highest-info products using the fine-tuned encoder.

---

## Repository layout

```
Digikala-Reviews-RAG/
├── Notebooks/
│   ├── digi.ipynb                     # preprocessing, summarization, TF-IDF, GLOT500 (zero/ft), analysis
│   ├── Question_Generator_Final.ipynb # Persian QA generation pipelines (LLMs + constraints)
│   └── Training_Models.ipynb          # fine-tuning GLOT500 dual-encoder (Sentence-Transformers)
├── Data/
│   ├── highest_3000_info_data.xlsx
│   └── highest_3000_info_data_QA.xlsx
├── Results/                           # produced by notebooks (example files)
│   ├── tfidf_top3_evidence.xlsx
│   ├── tfidf_top6_eval_results.xlsx
│   ├── zero_glot_top6_contexts.xlsx
│   ├── finetuned_glot_top3_contexts.xlsx
│   ├── finetuned_glot_top6_contexts.xlsx
│   ├── glot500_eval.xlsx
│   ├── glot500_finetuned_eval.xlsx
│   ├── models_comparison_summary.xlsx
│   ├── per_question_gold_rank_score.xlsx
│   ├── qa_dup_similarity_report.xlsx
│   ├── gold_rank_per_question.png
│   ├── gold_score_per_question.png
│   └── gold_score_per_question_full_without_coverage.png
└── NLP_HW2_Report_Final.pdf           # Persian report summarizing the project and findings
```

---

## Data requirements

Place the original Digikala CSVs at repo root (or update paths in the notebooks):

* `digikala-products.csv` (must include `id`, `title_fa`, `Brand`, `Category1`, `Category2`, `Price`, `min_price_last_month`, …)
* `digikala-comments.csv` (must include `product_id`, `rate`, `likes`, `dislikes`, `body`, `advantages`, `disadvantages`, …)

**Notes**

* The pipeline **drops products with zero comments**, and can optionally keep only products with ≥4 comments.&#x20;
* All text normalization is performed with **Hazm**; URLs and ZWNJ are stripped.

---

## Environment & installation

The notebooks are **Colab-friendly**. Minimal deps:

```bash
pip install pandas numpy hazm tqdm openpyxl scikit-learn matplotlib transformers sentencepiece joblib sentence-transformers
```

Optional (when used in the notebooks):

```bash
pip install ace_tools summa fuzzywuzzy bitsandbytes accelerate
```

> GPU is recommended for embedding models (GLOT500, Persian LLaMA/Mistral/Mind) and for fine-tuning; TF-IDF runs fine on CPU.

---

## Quickstart (typical workflow)

1. **Preprocess + Summarize** → `Notebooks/digi.ipynb`

* Creates random subsets for exploration, plots comment/rating distributions.
* Removes **zero-comment** products and saves `cleaned-products.csv`.
* Aggregates comments per product and builds:

  * `comments_summary`: TF-IDF extractive summary over up to 400 comments.
  * `pros_sample`, `cons_sample`: cleaned bullet lists (robust parser for broken bracketed lists).
  * `full_summary`: templated Persian summary with title, category, price, votes/likes/dislikes + pros/cons.
* Exports:

  * `*_product_summaries(.csv/.xlsx)` (subset and full)
  * `final_data(.csv/.xlsx)` and `highest_{N}_info_data(.csv/.xlsx)` (sorted by info content)

2. **Build retrievers & evaluate**

* **TF-IDF retrieval**
  Creates a TF-IDF over `full_summary` with `(1,2)` n-grams and sublinear TF; returns Top-K with **cosine similarity** + best evidence sentence per candidate. Saves:

  * `tfidf_top6_eval_results.xlsx`
  * `tfidf_top3_evidence.xlsx`

* **GLOT500 (zero-shot)**
  Encodes questions and long summaries (sliding-window, length 512, stride 128), L2-normalizes, optional **PCA-whitening** (256d), retrieves Top-K. Saves:

  * `zero_glot_top6_contexts.xlsx`
  * `glot500_eval.xlsx` (oracle vs retrieval modes)

* **GLOT500 (fine-tuned)**
  Fine-tuning with **MultipleNegativesRankingLoss** (Sentence-Transformers).
  Positives are `(question, product full_summary)` pairs from **auto-generated QA**. We freeze early transformer layers, pool by mean, and re-embed full corpus with sliding-window + L2. Saves:

  * `finetuned_glot_top6_contexts.xlsx`, `glot500_finetuned_eval.xlsx`
  * Comparative plots & per-question tables under `Results/`.

3. **Question generation** → `Notebooks/Question_Generator_Final.ipynb`

* Several paths tested: mT5 Persian QG, **PersianMind**, **Persian LLaMA**, **Persian Mistral**, Gemini / GPT-style prompts with strict formatting.
* A robust **system prompt** enforces that **the product name must appear in every Q and A** and forbids vague “این محصول/کالا…”.
* Optional **NER masking** of entities before QG; optional translation-based EN QG path.
* Output is stitched into `qa_pairs` blocks and stored back to CSV/XLSX used by the fine-tuning step.

4. **Analysis & visualization** (in `digi.ipynb`)

* Computes **Hit\@K** and **MRR** for different retrievers; produces per-question **gold rank** and **gold score** plots (`gold_rank_per_question.png`, `gold_score_per_question.png`) and a workbook `models_comparison_summary.xlsx`.
* Examines how **lexical overlap** between the question and evidence correlates with success, comparing TF-IDF vs embedding models.

5. **Duplicate detection (bonus)**

* On `highest_3000_info_data_QA.csv`, detects duplicate QA rows by **cosine similarity** on the fine-tuned encoder embeddings; writes thresholds, coverage, FP/FN sheets to `qa_dup_similarity_report.xlsx`.

---

## Key results (short)

From the Persian report (human evaluation; 50 questions × 3 candidates per model, lower rank is better):

| Model                    | mean rank | median | n   |
| ------------------------ | --------- | ------ | --- |
| **TF-IDF**               | **2.727** | 3.000  | 150 |
| **GLOT500 – zero-shot**  | 4.587     | 5.000  | 150 |
| **GLOT500 – fine-tuned** | 3.453     | 4.000  | 150 |

TF-IDF is a very strong baseline when lexical overlap is high; fine-tuning GLOT500 **significantly improves over zero-shot** and narrows the gap to TF-IDF.&#x20;

**Automatic metrics used:** Hit\@6 and MRR\@6 for retrieval, plus per-question evidence extraction for qualitative auditing.&#x20;

---

## Minimal code snippets

**TF-IDF retrieval (Top-K):**

```python
import joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("highest_3000_info_data.csv").drop_duplicates("product_id")
ids  = df["product_id"].astype(str).tolist()
docs = df["full_summary"].astype(str).tolist()

vect = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, norm="l2")
X = vect.fit_transform(docs)

def retrieve(q, k=6):
    s = cosine_similarity(vect.transform([q]), X)[0]
    top = s.argsort()[-k:][::-1]
    return [{"rank":i+1,"score":float(s[j]),"product_id":ids[j],"context":docs[j]}
            for i,j in enumerate(top)]
```

**Fine-tuned GLOT500 (projected):**

```python
import numpy as np, joblib
from sklearn.preprocessing import normalize
# load pca + ctx_proj from notebook outputs, then:
def retrieve_ft(question, top_k=6):
    qv = embed_question_proj(question)     # (1,d)  from notebook helper
    sims = (qv @ ctx_proj.T)[0]
    idxs = np.argsort(sims)[-top_k:][::-1]
    return [{"rank":i+1,"score":float(sims[j]),"product_id":product_ids[j],"context":full_summaries[j]}
            for i,j in enumerate(idxs)]
```

---

## Notebook guide (what each one does)

* **`Notebooks/digi.ipynb`**

  * Data exploration and filtering (drop zero-comment products; optional ≥4 comments).&#x20;
  * TF-IDF extractive summaries + templated `full_summary`.
  * TF-IDF retrieval & evidence extraction; GLOT500 zero-shot retrieval (+ PCA whitening).
  * Evaluation exports and plots; duplicate-QA analysis.

* **`Notebooks/Question_Generator_Final.ipynb`**

  * Persian QA generation via several LLMs/seq2seq models with strong formatting constraints.
  * NER masking flow; Gemini/GPT recipes (replace keys with your own!) and strict JSON / block output.
  * Produces `*_QA*.csv/.xlsx` used by fine-tuning / evaluation.

* **`Notebooks/Training_Models.ipynb`**

  * Fine-tunes **cis-lmu/glot500-base** with Sentence-Transformers (MultipleNegativesRankingLoss).
  * Sliding-window doc encoding, L2 normalize, and optional whitening; re-runs evaluation.

---

## Main outputs (look under `Results/`)

* **TF-IDF:** `tfidf_top3_evidence.xlsx`, `tfidf_top6_eval_results.xlsx`
* **Zero-shot GLOT500:** `zero_glot_top6_contexts.xlsx`, `glot500_eval.xlsx`
* **Fine-tuned GLOT500:** `finetuned_glot_top3_contexts.xlsx`, `finetuned_glot_top6_contexts.xlsx`, `glot500_finetuned_eval.xlsx`
* **Comparisons & viz:** `models_comparison_summary.xlsx`, `per_question_gold_rank_score.xlsx`, `gold_rank_per_question.png`, `gold_score_per_question.png`
* **Duplicates:** `qa_dup_similarity_report.xlsx`

---

## Notes & gotchas

* **Secrets:** The example notebooks show API usage for external LLMs. **Do not hard-code keys**; use environment variables instead.
* **Large HF models:** Some Persian LLMs need extra RAM/GPU. Prefer 4-bit loading (`bitsandbytes`) when available.
* **Text noise:** Pros/cons in Digikala often arrive as messy bracketed strings; our parser in `digi.ipynb` normalizes them and drops non-informative tokens (e.g., «ندارد», «هیچی»).
* **Weak summary filter:** Extremely short, low-signal summaries (or summaries with price==0 and no pros/cons) are removed before training/eval.

---

## Repro checklist

1. Put the two raw CSVs at repo root.
2. Open **`Notebooks/digi.ipynb`** and run cells (subset creation → cleaning → summaries → `final_data*.csv/xlsx`).
3. Run **TF-IDF** and **GLOT500 (zero-shot)** sections → artifacts land in `Results/`.
4. Open **`Notebooks/Question_Generator_Final.ipynb`** → generate Persian QA blocks → write `*_QA.csv/xlsx`.
5. Open **`Notebooks/Training_Models.ipynb`** → fine-tune GLOT500 and re-evaluate.
6. Inspect plots/workbooks under `Results/` and (optional) export for human eval (Label Studio template in report).&#x20;

---

## Report

A Persian write-up of the full pipeline, metrics, and human-eval protocol is in **`NLP_HW2_Report_Final.pdf`**.
Highlights: data cleaning strategy, strong TF-IDF baseline under high lexical overlap, zero-shot vs fine-tuned GLOT500 comparison, and Label-Studio ranking analysis.&#x20;

---

## Acknowledgments

* **cis-lmu/glot500-base** for multilingual sentence representations.
* Hazm for Persian normalization.
* Sentence-Transformers for dual-encoder fine-tuning.

---

## License

This project is for academic use. If you build upon it, please cite the report and link back to this repository.
