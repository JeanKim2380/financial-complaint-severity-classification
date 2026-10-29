# Financial Complaint and Severity Classification with Multi-Task Transformers

Do related tasks help a Transformer detect customer complaints in financial text and judge how severe they are? This project compares **single-task (STL)** and **multi-task (MTL)** learning on the FINCORP corpus of online financial complaints and opinions.

## Tasks

| Task | Abbreviation | Labels |
|---|---|---|
| Complaint identification | CI | complaint vs non-complaint |
| Complaint severity | CS | 4 severity levels |
| Emotion recognition | ER | auxiliary task |
| Sentiment analysis | SA | auxiliary task |

CI and CS are each treated as the **primary task**. Every combination of the other tasks is added as an auxiliary task, giving 8 configurations per primary task.

## Model

- **Shared Transformer encoder:** RoBERTa-base, DistilBERT-base-uncased or FinBERT (ProsusAI)
- **A separate BiGRU classification head for each task**, on top of the encoder's token representations
- The losses from all active tasks are summed. The severity loss is **class-weighted** (balanced weights) because the severity levels are imbalanced.
- AdamW with separate learning rates for the encoder (5e-5) and the heads (1e-3), 3 epochs, batch size 32, max sequence length 50, and mixed precision on GPU

## Evaluation design

- Stratified 60 / 10 / 30 train / validation / test split, fixed by a split seed
- Each configuration trained with **3 random seeds** (40, 41, 42), with macro-F1 as the primary metric
- For the test set, the seed with the best **validation** score was used for each configuration, so test data played no part in model selection
- Mean ± standard deviation and paired significance tests against the single-task baseline, computed across seeds on the validation set

## Results (test set, macro-F1)

| Encoder | CI: STL | CI: best MTL | CS: STL | CS: best MTL |
|---|---|---|---|---|
| RoBERTa | **0.962** | 0.960 (CI+ER) | 0.342 | **0.505** (CS+CI+ER+SA) |
| DistilBERT | 0.956 | **0.959** (CI+SA) | 0.477 | **0.530** (CS+SA) |
| FinBERT | **0.962** | 0.962 (CI+SA+CS) | 0.432 | **0.512** (CS+ER) |

The full results for all 48 configurations are in [`results/test_results.csv`](results/test_results.csv), and the per-seed validation results are in [`results/validation_results.csv`](results/validation_results.csv).

**Findings**
- **Complaint identification is already close to its ceiling** (macro-F1 about 0.95–0.96) with single-task learning, and auxiliary tasks make no meaningful difference.
- **Severity classification benefits clearly from multi-task learning.** The best MTL setup beats the single-task baseline for all three encoders, by 5 to 16 macro-F1 points. This suggests that the auxiliary tasks give the harder, imbalanced severity task useful extra supervision.
- FinBERT, which was pre-trained on financial text, does not clearly outperform the general-purpose encoders on either task.

## Limitations

- Test scores come from a single seed per configuration (the one selected on validation), so small differences between configurations, especially on CI, are within seed-to-seed noise.
- Severity macro-F1 remains modest (about 0.5), and the 50-token limit truncates longer posts.

## How to run

1. Download `FINCORP.csv` from the dataset authors' repository: <https://github.com/RohanBh23/FINCORP>. It is not redistributed here.
2. The notebook was written for Google Colab with a GPU. Put `FINCORP.csv` in a Drive folder and update the `%cd` path in the data-loading cell, or remove the Drive cells to run it locally.
3. Install the dependencies and run the notebook:
   ```bash
   pip install -r requirements.txt
   jupyter notebook financial_complaint_mtl.ipynb
   ```
   Running the full grid trains 144 models (48 configurations × 3 seeds), which takes about 2 hours on a single GPU.

## Dataset reference

Singh et al., *Complaint and Severity Identification from Online Financial Content* (IEEE). Dataset and code: <https://github.com/RohanBh23/FINCORP>.

## Tools

Python · PyTorch · Hugging Face Transformers · scikit-learn · pandas · NumPy · SciPy · Matplotlib · seaborn
