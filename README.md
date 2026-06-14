# AI-Driven Insights for Individualized Learning

MSc AI Dissertation, University of West London, 2024 | Grade: 75 - Distinction

## Research Questions
1. How does enrollment age impact academic performance + retention?
2. How does gender correlate with units credited/approved?
3. How do demographics influence learner analytics effectiveness?
4. Can predictive analytics forecast dropout for early intervention?

## Methods
- **Models**: Linear Regression, Decision Trees
- **Features**: age, gender, analytics utilization 
- **Targets**: curricular units credited, dropout/graduate
- **Dataset**: N= higher education students 

## Project Structure
- ├── projectms.ipynb       # Main Jupyter/Colab notebook containing PySpark EDA and Scikit-Learn models
- ├── README.md             # Repository summary and documentation
- └── requirements.txt      # Required Python dependencies

## Pipeline Structure:
1. Data Ingestion & Profiling: Handled via PySpark to verify structural distributions and parse multi-feature schemas.
2. Exploratory Data Analysis (EDA): Feature isolation via target variable cross-tabulation (evaluating graduation vs. dropout rates against core categorical segments like gender and financial standing).
3. Data Quality Optimization: Addressed via feature scaling (StandardScaler), missingness profiling (missingno), and collinearity reduction.
4. Predictive Modeling: Built using comparative architectures:
5. Random Forest Classifier (Ensemble bagging approach)
6. Logistic Regression Classifier (Parametric probability baseline)
7. Hyperparameter Tuning & Evaluation: Performance optimization evaluated meticulously through Classification Reports before and after hyperparameter tuning.

## Impact
Provides evidence for early-warning systems in universities to reduce dropout rates.

## Full Thesis
See Mehrunnisa Shaikh Dissertation.docx.pdf
