# Structural Diversity Drives Disruptive Scientific Innovation

This repository contains the replication code for the paper:

> **Structural Diversity Drives Disruptive Scientific Innovation**  
> Authors: Yichun Peng, Saike He, Peijie Zhang, et al.

## 📂 Data Availability
### 1. Processed AMiner Dataset (Ready for Replication)
The processed dataset used for the **Robustness Checks (Figure 3)** and **Quasi-Natural Experiment (Figure 4)** is available on Figshare.
👉 **Download Data Here:** [https://doi.org/10.6084/m9.figshare.31288000](https://doi.org/10.6084/m9.figshare.31288000)
**Instructions:**
1. Download the `.csv` file from the Figshare link above.
2. Place the file in the **root directory** of this repository.

### 2. OpenAlex Dataset (Source Code Provided)
The large-scale bibliographic data (Parquet format) used for the main regression analysis (Tables 1 & 2) and mediation analysis (Figure 5) exceeds storage limits (>1.2 TB) and cannot be hosted here.
* The raw data can be accessed via [OpenAlex](https://openalex.org/).
* The Python scripts (`01`–`04`) are provided for **methodological transparency**, allowing readers to inspect the variable construction and regression specifications.

## 🚀 Repository Structure
### 1. Robustness Checks & Visualization (Executable)
* **`Robustness_Checks_PSM_and_NSF.ipynb`**: 
  * A Jupyter Notebook that reproduces the **Propensity Score Matching (PSM)** analysis (Figure 3) and the **Quasi-Natural Experiment** (Figure 4).
  * *Requires:* The AMiner CSV file from Figshare.
### 2. Main Regression Analysis (Methodological Reference)
* **`01_reproduce_table1.py`**: Reproduces the OLS regression for the main effect of Structural Diversity (SD) on Innovation (CD Index).
* **`02_reproduce_table2.py`**: Reproduces the interaction effect analysis between SD and Team Size.
* *Note:* These scripts require the full OpenAlex parquet dataset.
### 3. Mediation Analysis (Methodological Reference)
* **`03_reproduce_mediation_step1.py`**: Tests Path A (SD -> Disciplinary Integration).
* **`04_reproduce_mediation_step2.py`**: Tests Path B & C (DI -> CD Index) to calculate the indirect effect.
