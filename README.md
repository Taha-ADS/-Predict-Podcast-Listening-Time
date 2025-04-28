# 📚 Podcast Listening Time Prediction

This repository contains a full machine learning pipeline to predict **podcast episode listening time** based on episode metadata, show popularity, sentiment, and textual features like episode titles and podcast names.

## 🛠️ Features & Pipeline

- **Data Preprocessing**
  - Extract numerical features (e.g., episode number from titles).
  - Frequency encoding for podcast names.
  - Label encoding of categorical variables (e.g., genre, publication time, sentiment).

- **Text Feature Engineering**
  - TF-IDF vectorization on both `Episode_Title` and `Podcast_Name`.
  - Combined with dense features into a unified sparse matrix for model training.

- **Baseline Models**
  - Ridge Regression using only TF-IDF on Episode Titles.
  - Histogram-Based Gradient Boosting (HGB) with basic tabular and text features.
  - Stacked ensemble of Ridge and HGB models to improve performance.

- **Advanced Modeling**
  - Hyperparameter tuning with `RandomizedSearchCV` on LightGBM (LGBMRegressor).
  - Manual early stopping with LightGBM callback API to avoid API conflicts inside cross-validation.
  - Careful model retraining and validation to achieve RMSE improvements.

## 🧪 Evaluation Metrics

- Primary evaluation metric: **Root Mean Squared Error (RMSE)**.
- Achieved consistent validation RMSE improvements:
  - RidgeCV Baseline RMSE: **13.33**
  - Tuned HGB RMSE: **12.97**
  - Stacked Ridge+HGB RMSE: **12.97**
  - LightGBM Tuned Model: aiming for RMSE in the **lower 12s**.

## 📈 Next Steps (Planned)

- Further tuning TF-IDF (larger vocab, trigram features).
- Model stacking or blending with neural networks.
- Full cross-validation (instead of simple train/val splits).
- Deployment-ready notebook and inference pipeline.

---

# 📦 Project Structure

```bash
.
├── data/             # Raw and processed CSVs
├── notebooks/        # Experimentation notebooks
├── src/              # Scripts for feature engineering and modeling
├── models/           # Saved model artifacts
└── README.md         # This file
```

---

# 🤝 Contributing

Contributions, feedback, and ideas for model improvements are welcome!  
Feel free to open an issue or submit a pull request.
