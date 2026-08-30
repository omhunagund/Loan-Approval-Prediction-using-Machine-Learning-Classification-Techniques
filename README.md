# Loan Approval Prediction using Machine Learning Classification Techniques

##  Project Overview
This project focuses on predicting whether a loan application will be approved or rejected using machine learning classification techniques. The model analyzes applicant-related financial and socio-economic factors such as income, credit history, employment status, loan amount, and property area to identify patterns influencing loan approval decisions.

The project includes data cleaning, preprocessing, exploratory data analysis, classification model building, performance evaluation, and business interpretation.

---

##  Objective
The objective of this project is to develop classification models that predict loan approval outcomes based on socio-economic and financial factors and to identify the key factors that influence loan approval decisions.

---

##  Tools & Technologies
- **Python**
- **Jupyter Notebook**
- **Pandas** – Data manipulation and preprocessing
- **NumPy** – Numerical operations
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical visualization
- **Scikit-learn** – Machine learning and model evaluation

### Machine Learning Models
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC Score

---

##  Dataset
The project uses a **Loan Prediction Dataset** containing information about loan applicants and their financial and demographic characteristics.

### Main Features
- **Gender** – Applicant gender
- **Married** – Marital status
- **Dependents** – Number of dependents
- **Education** – Applicant education level
- **Self_Employed** – Employment status
- **ApplicantIncome** – Applicant's income
- **CoapplicantIncome** – Co-applicant's income
- **LoanAmount** – Requested loan amount
- **Loan_Amount_Term** – Loan repayment term
- **Credit_History** – Applicant's credit history
- **Property_Area** – Area category of the property
- **Loan_Status** – Target variable

### Target Variable
```text
Y → Loan Approved
N → Loan Not Approved
```

---

##  Project Workflow

```text
Dataset
  ↓
Data Import
  ↓
Initial Data Exploration
  ↓
Data Cleaning
  ↓
Categorical Encoding
  ↓
Feature Scaling
  ↓
Exploratory Data Analysis
  ↓
Train-Test Split
  ↓
Model Building
  ↓
Model Evaluation
  ↓
Confusion Matrix & ROC-AUC
  ↓
Model Comparison
  ↓
Business Insights
```

---

##  Data Cleaning & Preprocessing
The following preprocessing steps were performed:
- Examined the dataset structure, data types, shape, and missing values.
- Handled missing values using appropriate techniques such as median imputation for numerical features and mode imputation for categorical features.
- Converted categorical variables into numerical representations using Label Encoding and One-Hot Encoding.
- Converted the `3+` value in the `Dependents` column into a numerical representation (`3`).
- Created an `Income_Group` column for income-based EDA and removed it before model training because it was intended only for visualization.
- Applied `StandardScaler` to scale the numerical input features before model training.
- Used a stratified train-test split to maintain the class distribution between training and testing data.

---

##  Exploratory Data Analysis
The EDA was performed based on the project requirements and focused on understanding loan approval patterns.

**1. Approval Rate by Gender**
The approval distribution was analyzed across male and female applicants.
*Insight:* Loan approval rates were relatively similar across genders, suggesting that gender alone is not a strong predictor of loan approval.

**2. Approval Rate by Income**
Applicants were grouped into income categories to analyze approval patterns across different income levels.
*Insight:* Higher-income groups generally showed somewhat better approval rates, although income alone did not determine the final approval outcome.

**3. Approval Rate by Employment Status**
Approval rates were compared between self-employed and non-self-employed applicants.
*Insight:* Employment status showed some variation in approval rates, indicating a moderate relationship with loan approval decisions.

**4. Correlation Matrix**
A correlation matrix was used to examine relationships between numerical variables.
*Insight:* `Credit_History` showed the strongest positive relationship with `Loan_Status`, highlighting its importance in predicting loan approval.

**5. Boxplots for Numerical Features**
Boxplots were used to compare numerical variables such as applicant income and loan amount across loan approval outcomes.
*Insight:* Approved and rejected applicants showed some differences in their distributions, but substantial overlap was present between the groups.

---

##  Model Building
Four classification algorithms were implemented:

**Logistic Regression**
Used as a baseline classification model for predicting binary loan approval outcomes.

**Decision Tree Classifier**
Used to capture non-linear relationships and decision patterns within the applicant data.

**Random Forest Classifier**
Used as an ensemble model capable of capturing more complex relationships while reducing the variance associated with individual decision trees.

**Support Vector Machine (SVM)**
Used to identify an effective decision boundary between approved and rejected loan applications.

---

##  Model Evaluation
The models were evaluated using:
- **Accuracy** – Measures the proportion of correctly classified applications.
- **Precision** – Measures how many applications predicted as approved were actually approved.
- **Recall** – Measures how many actual approved applications were correctly identified.
- **F1 Score** – Provides a balance between precision and recall.
- **Confusion Matrix** – Shows correct and incorrect classifications for each class.
- **ROC-AUC** – Measures the model's ability to distinguish between approved and rejected applications.

### Model Performance
The Logistic Regression model achieved the highest accuracy among the reported models:

| Model | Accuracy |
|---|---|
| Logistic Regression | 86.17% |
| SVM | 85.36% |
| Decision Tree | 75.60% |
| Random Forest | 82.92% |

> The final model comparison was based on the evaluation metrics generated in the notebook.

---

##  Key Insights
- Credit history is one of the strongest factors associated with loan approval.
- Income level has an observable influence on approval outcomes, although it is not sufficient by itself to determine approval.
- Employment status shows moderate variation in approval patterns.
- Loan amount and applicant financial characteristics contribute to the overall decision-making process.
- Logistic Regression achieved the highest reported accuracy of 86.17% among the models evaluated.
- The confusion matrix demonstrated that the selected model correctly classified the majority of loan applications, while some misclassifications remained.
- ROC-AUC analysis provided an additional measure of the model's ability to separate approved and rejected applications.

---

##  How a Bank Could Use This Model
In a practical banking environment, the model can be integrated into the loan processing workflow as a decision-support system.

When a customer submits a loan application, relevant information such as income, credit history, employment status, loan amount, and property area can be passed to the model. The model can then generate a predicted approval outcome or probability.

Applications predicted as lower risk can be processed more efficiently, while applications with higher risk can be flagged for further assessment by credit officers.

The model can therefore help:
- Reduce initial loan screening time
- Handle large volumes of applications
- Improve consistency in preliminary evaluation
- Support credit risk assessment
- Assist human decision-makers with data-driven insights

> The model should be treated as a decision-support tool rather than a complete replacement for human review, particularly for high-risk financial decisions.

---

##  Challenges Faced & Solutions

**1. Missing Values**
Several columns contained missing values.
*Solution:* Numerical missing values were handled using median values, while categorical missing values were handled using the mode.

**2. Categorical Features**
Machine learning algorithms require numerical input, while several dataset columns contained categorical values.
*Solution:* Label Encoding and One-Hot Encoding were used to convert categorical features into numerical representations.

**3. Dependents Contained "3+"**
The `Dependents` column contained the value `3+`, which could not be directly processed as a numeric feature.
*Solution:* The `3+` category was converted to the numerical value `3`.

**4. Temporary EDA Feature**
An `Income_Group` feature was created for income-based EDA using categorical labels such as Low, Mid, and High. These values caused an error during feature scaling.
*Solution:* The temporary EDA feature was removed before model preparation because it was intended only for visualization.

**5. Feature Scaling**
Algorithms such as Logistic Regression and SVM can benefit from features being on comparable scales.
*Solution:* `StandardScaler` was applied after ensuring that all model input features were numerical.

---

##  Project Structure

```text
Loan-Approval-Prediction-using-Machine-Learning-Classification-Techniques/
│
├── loan_prediction.csv
├── Loan_Approval_Prediction.ipynb
└── README.md
```

---

##  How to Run the Project

1. Clone the repository
```bash
   git clone https://github.com/omhunagund/Loan-Approval-Prediction-using-Machine-Learning-Classification-Techniques.git
```
2. Navigate to the project directory
```bash
   cd Loan-Approval-Prediction-using-Machine-Learning-Classification-Techniques
```
3. Install the required libraries
```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```
4. Launch Jupyter Notebook
```bash
   jupyter notebook
```
5. Open the notebook
   Open the `.ipynb` file and execute the cells sequentially.

---

##  Project Outcome
The project successfully developed and evaluated multiple machine learning classification models for predicting loan approval. After preprocessing, exploratory analysis, model training, and performance evaluation, Logistic Regression achieved the highest reported accuracy of 86.17%, followed by SVM with 85.36%.

The analysis also highlighted the importance of factors such as credit history, income, employment status, and loan-related characteristics in understanding loan approval outcomes.

---

##  Author
**Om Hunagund**
GitHub: [github.com/omhunagund](https://github.com/omhunagund)