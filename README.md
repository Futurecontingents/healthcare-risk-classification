# Healthcare Risk Classification Using Logistic Regression and Naive Bayes

This is a small interpretable machine learning project built around a public medical insurance dataset. I wanted to practise a complete and readable workflow: define a target, explore the data, train simple probabilistic classifiers, evaluate them carefully, and explain the mathematics without hiding behind complicated code.

The project is educational. It studies statistical relationships in the dataset and does not prove medical causation. It is not a clinical tool and should not be treated as medical advice.

## Project Overview

The original dataset contains insurance charges as a continuous variable. I turned this into a binary classification problem:

```text
high_cost = 1 if charges are above the 75th percentile
high_cost = 0 otherwise
```

The aim is to predict whether a record belongs to the high-cost group using interpretable probabilistic models:

- Logistic Regression
- Gaussian Naive Bayes

I deliberately kept the modelling simple because the main goal is understanding:
- probability;
- odds and log-odds;
- Bayes theorem;
- coefficient interpretation;
- evaluation metrics.

## Motivation

For an internship portfolio, I wanted a project that shows both mathematical thinking and practical coding. Healthcare data is a useful setting for this because it forces careful language: a model can identify statistical associations, but that is not the same as proving a medical cause.

One thing I focused on was being able to explain every part of the project in an interview. I avoided advanced models so that the notebook and report remain realistic for a second-year Computer Science and Artificial Intelligence student.

## Dataset Description

The dataset contains 1,338 insurance records with the following columns:

- `age`: age of the insurance beneficiary
- `sex`: recorded sex category
- `bmi`: body mass index
- `children`: number of covered children
- `smoker`: whether the person is recorded as a smoker
- `region`: residential region
- `charges`: individual medical insurance charges

The dataset file itself is not included in the repository.

Place:

```text
insurance.csv
```

in the project root before running the notebook.

## Methods

The notebook uses:

- a stratified train/test split;
- one-hot encoding for categorical variables;
- Logistic Regression for coefficient and odds interpretation;
- Gaussian Naive Bayes as a probabilistic baseline;
- confusion matrices and standard classification metrics.

The model features are:

```text
age, bmi, children, sex, smoker, region
```

The `charges` column is not used as a feature because it defines the target. Using it directly would leak the answer into the model.

## Mathematical Concepts

The report and notebook explain:

- sigmoid function;
- logit and log-odds;
- odds and odds multipliers;
- logistic regression coefficients;
- Bayes theorem;
- prior, evidence, and posterior probability;
- Naive Bayes conditional independence assumption;
- accuracy, precision, recall, and F1-score;
- why prediction is not the same as causation.

## File Structure

```text
.
├── README.md
├── requirements.txt
├── notebook/
│   └── healthcare_risk_classification.ipynb
├── report/
│   ├── healthcare_risk_report.tex
│   ├── healthcare_risk_report.pdf
│   └── figures/
├── outputs/
│   ├── figures/
│   └── tables/
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── visualization.py
└── interview_notes.md
```

## How To Run

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Launch the notebook:

```bash
jupyter notebook notebook/healthcare_risk_classification.ipynb
```

To execute the notebook from the command line:

```bash
jupyter nbconvert --to notebook --execute notebook/healthcare_risk_classification.ipynb --inplace
```

To compile the LaTeX report:

```bash
tectonic -X compile report/healthcare_risk_report.tex --outdir report
```

## Example Outputs

The notebook generates figures including:

- distribution of charges;
- distribution of `log(charges)`;
- high-cost class balance;
- high-cost rate by smoker status;
- high-cost rate by BMI category;
- high-cost rate by age group;
- charges vs age;
- charges vs BMI;
- numerical correlation matrix;
- confusion matrices for both models.

The model comparison table is saved at:

```text
outputs/tables/model_comparison.csv
```

In the final train/test split, Logistic Regression and Naive Bayes performed similarly. The exact values are available in the notebook and generated tables. I would not overstate the result because this is one dataset and one simple split.

## Simple Limitations

- The high-cost threshold is arbitrary and chosen for learning.
- The target is based on insurance cost, not a medical diagnosis.
- The dataset is relatively small and simplified.
- The analysis is observational.
- Correlation and model coefficients do not prove causality.
- Naive Bayes assumes conditional independence, which is probably unrealistic for some variables.
- The project uses one train/test split rather than a full validation study.

This project studies statistical relationships in the dataset and does not prove medical causation.

## Future Improvements

- Add cross-validation after mastering the basic train/test workflow.
- Compare different high-cost thresholds.
- Add a short fairness discussion around sensitive variables.
- Add probability calibration checks.
- Write a small model card describing intended and non-intended use.

## What I Learned

I learned that interpretable models still require careful explanation. Logistic regression is simple to code, but its coefficients only make sense if I understand odds and log-odds properly. Naive Bayes is also simple to train, but its independence assumption needs to be stated honestly.

One interesting observation was that smoking status separated the high-cost and not-high-cost groups quite strongly in this dataset. That is an association in the data, not proof of medical causation.

## CV-Ready Project Description

Built an interpretable healthcare risk classification project using Python, scikit-learn and LaTeX, comparing logistic regression and Naive Bayes on a public medical insurance dataset, with emphasis on probability interpretation, Bayes theorem, coefficient analysis and model evaluation.
