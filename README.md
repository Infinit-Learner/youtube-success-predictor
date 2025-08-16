# YouTube Success Predictor

A machine learning project aimed at predicting a video’s success **before upload**, using publicly available YouTube metadata. The goal goes beyond predictive accuracy — this project is structured to showcase *professional-level ML engineering practices*, including feature hygiene, reproducible pipelines, and model selection methodology.

> ⚙️ **Current Stage:** Baseline → tuned model complete  
> 🧠 **Project Goal:** Skill development and portfolio demonstration (ML Engineering focused)

---

## 🎯 Problem Statement

Can we estimate how well a YouTube video will perform *before it goes live*, using only information available to the creator at upload time (title, description, category, etc.) — *without relying on post-upload engagement features like likes or comments?*

---

## 📊 Dataset

- Source: **Trending YouTube Videos (US)** dataset (Kaggle)
- Type: Historical trending video snapshots with metadata  
- Target: `views` (modeled in log-space)  
- **Leakage Avoidance:** Explicit removal of all post-upload engagement variables (`likes`, `dislikes`, `comment_count`, `video_error_or_removed`, `trending_date`) during preprocessing.  
- ⚠️ **Dataset Bias:** Since this dataset contains *only videos that already trended*, the model learns to differentiate *degrees of trending success*, rather than predicting whether a random video will trend at all.

---

## 🔬 Workflow Overview

01 - Exploratory Data Analysis (EDA)
02 - Preprocessing & Feature Engineering
03 - Baseline Modeling
04 - Hyperparameter Tuning
05 - Final Evaluation on Unseen Hold-Out Data


Models tested: Linear Regression, Ridge, Lasso, Random Forest, XGBoost  
Metrics used: MSE

---

## 📁 Project Structure

data/
    raw/
    processed/
notebook/
    01-eda.ipynb
    02-preprocessing.ipynb
    03-modeling.ipynb
src/
(under development – pipeline modules coming next)
outputs/
    figures/
    models/



---

## 🚧 Roadmap / Next Steps

- 🔧 Modularize preprocessing + modeling logic into `/src/`
- 🧪 Create `main.py` runnable pipeline
- 📦 Serialize final model (`.pkl`) + build inference wrapper
- 🌍 Optional deployment via API or Streamlit interface

---

## 👤 About the Author

This project was built as part of a transition into professional machine learning engineering. Heavy emphasis was placed on *clarity, reproducibility, pipeline design*, and strong modelling fundamentals.  
Future iterations will introduce production-grade workflow, deployment, and broader dataset generalization.

---

## 📄 License

For educational use only. For commercial or enterprise reuse, please contact the author.