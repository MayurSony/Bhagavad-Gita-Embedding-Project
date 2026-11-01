# English-to-Sanskrit Bhagavad Gita Verse Retrieval Using Fine-Tuned Embeddings

## 1. Project Overview

This project develops a semantic retrieval system that retrieves relevant Sanskrit verses from the Bhagavad Gita using English translation queries.

A multilingual embedding model is fine-tuned using Sentence Transformers to improve cross-lingual semantic retrieval between English queries and Sanskrit verse passages.

## 2. Problem Statement

Finding a specific Sanskrit verse using an English translation can be challenging because the two languages use different vocabulary and grammatical structures.

This project explores multilingual embeddings and semantic search to retrieve the correct Sanskrit verse based on an English query.

## 3. Objectives

- Build an English-to-Sanskrit verse retrieval system.
- Fine-tune a multilingual embedding model for cross-lingual retrieval.
- Compare the base model against the fine-tuned model.
- Evaluate retrieval performance using Recall@K and Mean Reciprocal Rank (MRR).
- Analyze retrieval errors and identify potential improvements.

## 4. Technology Stack

- Python
- Google Colab
- PyTorch
- Sentence Transformers
- Hugging Face Datasets
- Pandas and NumPy
- Multilingual E5 embeddings
- Semantic similarity search

## 5. Dataset

The project uses the `OEvortex/Bhagavad_Gita` dataset.

The cleaned dataset contains 700 verse records with fields including chapter information, Sanskrit text, Hindi translation, and English translation.

| Dataset Split | Records | Percentage |
|---|---:|---:|
| Training | 490 | 70% |
| Validation | 70 | 10% |
| Testing | 140 | 20% |
| **Total** | **700** | **100%** |

The dataset was split using `random_state=42` for reproducibility.

## 6. Model and Methodology

### Base Model

`intfloat/multilingual-e5-small`

The model generates 384-dimensional embeddings and supports multilingual semantic representations.

### Fine-Tuning Configuration

The model was fine-tuned using Sentence Transformers with `MultipleNegativesRankingLoss`.

| Parameter | Value |
|---|---|
| Epochs | 2 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Embedding dimension | 384 |
| Maximum sequence length | 512 |
| Loss function | MultipleNegativesRankingLoss |

The E5 input prefixes were used:
- `query: ` for English queries
- `passage: ` for Sanskrit passages

### Retrieval Pipeline

1. Load and preprocess the Bhagavad Gita dataset.
2. Split the data into training, validation, and test sets.
3. Prepare English-query and Sanskrit-passage training pairs.
4. Fine-tune the multilingual embedding model.
5. Generate embeddings for Sanskrit verses.
6. Encode an English query into an embedding.
7. Calculate similarity between the query and verse embeddings.
8. Rank verses by similarity and retrieve the top-k results.
9. Evaluate retrieval performance on the held-out test set.

## 7. Evaluation Metrics

- **Recall@K:** Measures how often the correct verse appears among the top K retrieved results.
- **Mean Reciprocal Rank (MRR):** Measures how highly the correct verse is ranked, rewarding results that place it closer to the top.

## 8. Results

### Baseline vs. Fine-Tuned Model

| Metric | Base Model | Fine-Tuned Model |
|---|---:|---:|
| Recall@5 | 24.29% | 58.57% |

The fine-tuned model improved Recall@5 by **34.28 percentage points**, equivalent to approximately a **141.1% relative improvement** over the baseline.

### Fine-Tuned Model Performance

| Metric | Result |
|---|---:|
| Recall@1 | 37.14% |
| Recall@3 | 52.86% |
| Recall@5 | 58.57% |
| MRR | 0.4811 |

The fine-tuned model retrieved the correct verse within its top five results for 82 of the 140 test queries.

## 9. Error Analysis

A qualitative review of retrieval results found examples where the retrieved verses were related in theme or meaning to the target verse but did not match the exact verse identifier.

For example, verses concerning meditation, self-control, eternal nature, and sacrifice can be semantically related while expressing different teachings.

This highlights a challenge in semantic retrieval: conceptual similarity does not always guarantee an exact match.

These observations are preliminary and do not establish the cause of every retrieval error.

## 10. Limitations

- The dataset contains only 700 verse records.
- Some Sanskrit entries contain truncated text.
- Semantically related verses can be difficult to distinguish.
- Exact verse matching does not fully capture semantic relevance.
- Additional experiments are needed to determine how different training configurations and retrieval methods affect performance.

## 11. Future Improvements

- Clean and validate truncated Sanskrit passages.
- Expand the training data with additional examples.
- Experiment with different training hyperparameters and embedding models.
- Explore hybrid retrieval using semantic similarity and lexical matching.
- Conduct a more extensive error analysis.
- Evaluate alternative retrieval and reranking methods.

These improvements are proposed future work and have not been evaluated in the current experiment.

## 12. Project Files

- `sanskrit_english_embedding_finetuning.ipynb` — Notebook containing the training and evaluation workflow.
- `evaluation_results.csv` — Exported evaluation results.
- `sanskrit_english_e5_finetuned.zip` — Saved fine-tuned model.

*Note: The filenames listed above are descriptive. If your downloaded filenames differ, update this section to match your actual files.*

## 13. Conclusion

This project demonstrates how fine-tuning a multilingual embedding model can improve English-to-Sanskrit semantic retrieval for Bhagavad Gita verses.

The fine-tuned model achieved a Recall@5 of 58.57%, compared with 24.29% for the base model on the evaluated test set.

The results highlight the potential of cross-lingual embeddings for semantic retrieval while identifying data quality and exact verse matching as areas for further improvement.

## 14. Reproducibility

The training notebook documents the model fine-tuning and evaluation workflow.

Install the required Python libraries using `requirements.txt` before running the notebook. Model training may require additional time and computing resources.