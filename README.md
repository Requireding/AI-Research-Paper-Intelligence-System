# AI Research Paper Intelligence System

An NLP-powered semantic search and retrieval system designed to process, embed, and query massive collections of machine learning research papers. This system leverages state-of-the-art embedding models to map textual research abstracts into a continuous vector space, allowing for high-accuracy semantic search.

---

## 🚀 System Architecture & Workflow

The system is built sequentially in a modular Google Colab pipeline:

1. **Environment Setup & Data Ingestion**
   * Installs and configures the Hugging Face `datasets` library to securely stream a large corpus of machine learning data without heavy local overhead.
   * Loads the dataset `CShorten/ML-ArXiv-Papers` (specifically pulling the extensive `train` split containing over 117,000 research articles).

2. **Data Processing & Engineering**
   * Converts the raw dataset structure into a high-performance `pandas` DataFrame for manipulation.
   * Filters and isolates the critical feature sets: `title` and `abstract`.
   * Performs systematic structural checks (such as `.isnull().sum()`) to guarantee data integrity before sending text to the embedding layers.
   * Engineers a synthesized feature `paper_text` by structurally combining titles and abstracts to provide full contextual baseline tokens for downstream search algorithms.

3. **Vector Embeddings (Dense Retrieval Layer)**
   * Initializes the specialized `SentenceTransformer` framework.
   * Deploys the optimized `all-MiniLM-L6-v2` dense vector mapping model to encode textual data into high-dimensional numerical spaces, setting up the backend mechanics required for semantic similarity matching.

---

## 🛠️ Repository File Structure

* **`AI_Research_Paper_Intelligence_System.ipynb`**: The primary execution environment containing data pipeline loops, modeling layers, and data analysis utilities.
* **`.gitignore`**: Configured custom parameters ensuring large raw elements (like `ML-ArXiv-Papers.csv` or downloaded local `.safetensors` model weights) remain strictly untracked, protecting your storage limits on GitHub.
* **`LICENSE`**: Permissive open-source blueprint utilizing the global industry-standard MIT layout.
* **`README.md`**: Core system architecture, operational overview, and execution manifests.

---

## 💻 Tech Stack & Libraries Used

* **Language:** Python 3
* **Compute Environment:** Google Colab (GPU Accelerated - T4 Runtime)
* **Core Libraries:**
  * `datasets` (Hugging Face Dataset Ingestion API)
  * `pandas` (Structured Data Manipulation & Vector Operations)
  * `sentence-transformers` (State-of-the-Art Dense Text Embeddings)

---
**Student Self-Analysis & Observations**

## From my independent testing of the AI Research Paper Intelligence System, I discovered that:

**1. Robustness to Errors:** The dense vector model (all-MiniLM-L6-v2) handles heavily misspelled search queries surprisingly well, successfully extracting relevant machine learning abstracts even when keywords like "deep learning" were misspelled.

**2. Context Limitations:** When given an out-of-domain query (like a pizza recipe), the system still forces a match with the closest technical paper available because cosine similarity scores are relative. In a future production iteration, I would implement a minimum similarity score threshold (e.g., score > 0.40) to discard completely irrelevant queries.

## 📈 Next Steps & Planned Features
* [ ] Implement an approximate nearest neighbors (ANN) search index using **FAISS** or **ChromaDB** for instant querying across all 117k+ records.
* [ ] Build an interactive search interface utilizing **Gradio** or **Streamlit**.
* [ ] Integrate cross-encoder re-ranking to further improve retrieval precision.
