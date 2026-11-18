# NPS-Proxy-Prediction-from-Healthcare-Call-Center-Transcripts

# NPS-Proxy Prediction from Healthcare Call-Center Transcripts

End-to-end NLP pipeline that classifies call transcripts as **detractor / passive / promoter** using the AIxBlock 92k call-center dataset (Medicare, medical equipment, insurance).

- **Labeling:** two independent synthetic labelers (rule-based cue scoring and DistilBERT sentiment), with Cohen's kappa for agreement and optional human-labeled validation.
- **Features:** TF-IDF → TruncatedSVD (refit per fold to prevent leakage) plus structured features (length, ASR confidence, words per minute).
- **Models:** Logistic Regression, calibrated LinearSVC, and LightGBM.
- **Evaluation:** domain-holdout, hash-grouped stratified CV, and a cue-stripped human-label CV. Metrics are macro-F1, balanced accuracy, and Brier score.

> Labels are synthetic approximations of satisfaction, not true NPS scores.

**Stack:** Python, scikit-learn, LightGBM, HuggingFace Transformers, PyTorch, pandas, Google Colab
