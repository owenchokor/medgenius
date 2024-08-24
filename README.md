# **MEDGENIUS** : Feed Gigabytes of Clinical Guidelines for Accurate Medical Practice

### - by SNUH Hackathon team 2

![MEDGENIUS Explanation](./logo/MEdGenius.png)

We Leveraged RAG(Retrieval Augmentation Generation) technology to search and implement clinical guidelines for real use.
# How to Set Up

## 1. Create a Conda Virtual Environment

```bash
conda create -n my_env python=3.12
```

## 2. Activate the Virtual Environment
```bash
conda activate my_env
```

## 3. Clone the Git Repository
```bash
git clone https://github.com/owenchokor/medgenius.git
cd medgenius
```
## 4. Download Requirements
```bash
pip install -r requirements.txt
```
-----
# How to Use
```bash
python script.py --bucket_name snuh-data-team2 --pdf_source download --preprocess True
```
**arguments**
- bucket_name : Your S3 bucket name.
- pdf_source : You can either use your local pdf files(local) or files from the S3 bucket(download).
- preprocess : We implemented tabular data and image preprocessing to feed in the RAG model. Pass --preprocess True to enable preprocessing, and False to use only plaintext.
  
The diagram below shows our preprocessing scheme.
![Preprocessing Scheme](./logo/preprocessing_scheme.png)

## Target Users
- **General Practitioners and Specialists at Primary Hospitals**
  - MEDGENIUS is designed primarily for general practitioners and specialists at Primary Hospitals.

## Key Features
- **Providing Medical Standards**
  - The model offers comprehensive medical standards for healthcare professionals.
- **Quick Search Capability**
  - Enables quick searches for information outside of their specialty or for details they may have forgotten.

## Expected Benefits
- **Time Efficiency and Accuracy Improvement**
  - Saves time while enhancing the accuracy of diagnosis and treatment.
- **Patient-Centered Care**
  - Allows for personalized care tailored to each patient's specific needs, rather than standardized, one-size-fits-all treatments.

-----
# Specific Implementation Details
- **Tabular Data Embedding**
  - Clinical guidelines enclude a bunch of tabular data.
  - Camelot(https://github.com/camelot-dev/camelot) can extract tabular data in the form of DataFrame from the original raw data.
  - We fed the RAG model with Natural Language explanation of Camelot-extracted DataFrame.
- **Image Data Embedding**
  - We fed the RAG model with zero-shot image-to-text inference of the Anthropic Claude 3.5 Sonnet(https://www.anthropic.com/news/claude-3-5-sonnet) API.
  - We implemented a module that can show the context of QR code images.

----

