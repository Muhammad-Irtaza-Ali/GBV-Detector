# Urdu GBV Text Classifier

A Streamlit interface that loads a Hugging Face sequence-classification model and displays predictions for Urdu text.

**Category:** NLP / classification prototype

## Start here

Install `requirements.txt`, then run `streamlit run app.py`. The application downloads `muhammadirtazaali/gbv-bert-urdu` through Transformers, requiring internet access and sufficient memory. Model outputs and confidence scores are not verified safety assessments; evaluate labels, false positives, and false negatives before real-world use.

## Repository map

- [`.devcontainer/`](./.devcontainer)
- [`.gitignore`](./.gitignore)
- [`Copy_of_GBV_detect_project.ipynb`](./Copy_of_GBV_detect_project.ipynb)
- [`GBV_BERT/`](./GBV_BERT)
- [`GBV_detect_Urdu.json`](./GBV_detect_Urdu.json)
- [`app.py`](./app.py)
- [`requirements.txt`](./requirements.txt)

## Review and reproducibility

This guide describes the repository structure and documented implementation. It does not certify a successful installation, a passing test suite, production readiness, or claimed model accuracy. Keep individual contributions, upstream credits, data sources, and verified results with the relevant implementation.
