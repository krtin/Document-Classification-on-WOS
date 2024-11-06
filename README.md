# Document-Classification-on-WOS
https://paperswithcode.com/sota/document-classification-on-wos-5736

## Setup
You should have poetry and then do the following

```bash
poetry install
poetry shell
```

## Initial Thoughts
The dataset is a hierarchical classification dataset

**How to Model the output**
- Treat it as multi-label classification problem, where each document can have the all the labels from the hierarchy
- or treat this as a multi-class classification problem, where each unique label path is a class (this would make classes sparse and correlated)
- or we can do hierarchical classification, where we first predict the top level label, then the second level label, and so on

**Models to try**
- I can try classical models like SVM, Random Forest, etc maybe in a hierarchical fashion (some of them have been tried here https://arxiv.org/pdf/1709.08267v2), but I can still try to get a feel of the problem
- Other thing could be simply try hierarchical classification with transformer models like BERT
- Lastly, I can try prompting techiniques with LLMs I do not want to try SLMs at this stage, because I just want something that works with LLMs to form a zero or few shot baseline.

I should now begin with some EDA to understand the dataset better

## EDA
Dataset Downloaded From: https://data.mendeley.com/datasets/9rw3vkcfy4/6

Overall all classes are relatively we represented, Biochemistry is approximately 2x the size of other classes

| Domain | Area | Count |
|--------|------|-------|
| ECE | Digital control | 426 |
| ECE | Electricity | 447 |
| ECE | Operational amplifier | 419 |
| Psychology | Attention | 416 |
| Psychology | Child abuse | 404 |
| Psychology | Depression | 380 |
| Psychology | Social cognition | 397 |
| Biochemistry | Immunology | 652 |
| Biochemistry | Molecular biology | 746 |
| Biochemistry | Northern blotting | 699 |
| Biochemistry | Polymerase chain reaction | 750 |

## Classical Models

How to run:

```bash
python classical_models.py
```
Tried the text cleaning function https://arxiv.org/pdf/1709.08267v2, it had little impact on performance with classical models so default to not using it

- I am training 4 classifiers, one for categorizing the 3 main subjects and then 3 classifiers for topics under each subject.
- Between SVM and Random Forest, RD consistently performed marginally better than SVM

### Main Subject Classifier

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 1     | 0.97      | 0.98   | 0.97     | 249     |
| 2     | 0.97      | 0.91   | 0.94     | 324     |
| 6     | 0.96      | 0.99   | 0.97     | 574     |

2: Psychology
6: Biochemistry
1: ECE

**Accuracy**: 0.96 (1147 samples)

**Macro Avg**:  
- Precision: 0.97  
- Recall: 0.96  
- F1-Score: 0.96  
- Support: 1147  

**Weighted Avg**:  
- Precision: 0.96  
- Recall: 0.96  
- F1-Score: 0.96  
- Support: 1147  

**Test Accuracy**: 0.9643

Overall the accuracy is quite good, with the exception of 'Psychology' which has a lower recall compared to other 2
It seems at least for the top-level classification a classical model is good enough and cheap.


### Classifier for ECE

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.90      | 0.99   | 0.94     | 85      |
| 1     | 0.00      | 0.00   | 0.00     | 1       |
| 4     | 0.99      | 0.93   | 0.96     | 86      |
| 15    | 0.92      | 0.99   | 0.95     | 72      |
| 17    | 0.00      | 0.00   | 0.00     | 7       |

**Accuracy**: 0.94 (251 samples)

**Macro Avg**:  
- Precision: 0.56  
- Recall: 0.58  
- F1-Score: 0.57  
- Support: 251  

**Weighted Avg**:  
- Precision: 0.91  
- Recall: 0.94  
- F1-Score: 0.92  
- Support: 251  

**Test Accuracy**: 0.9363

### Classifier for Psychology

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.00      | 0.00   | 0.00     | 1       |
| 1     | 0.94      | 0.86   | 0.90     | 87      |
| 3     | 0.00      | 0.00   | 0.00     | 6       |
| 4     | 0.00      | 0.00   | 0.00     | 1       |
| 7     | 0.74      | 0.81   | 0.77     | 63      |
| 14    | 0.90      | 0.89   | 0.90     | 82      |
| 17    | 0.81      | 0.94   | 0.87     | 63      |

**Accuracy**: 0.85 (303 samples)

**Macro Avg**:  
- Precision: 0.48  
- Recall: 0.50  
- F1-Score: 0.49  
- Support: 303  

**Weighted Avg**:  
- Precision: 0.83  
- Recall: 0.85  
- F1-Score: 0.84  
- Support: 303  

**Test Accuracy**: 0.8515

### Classifier for Biochemistry

## Test Results

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.75      | 0.82   | 0.78     | 141     |
| 1     | 0.00      | 0.00   | 0.00     | 3       |
| 3     | 0.89      | 0.85   | 0.87     | 140     |
| 4     | 0.00      | 0.00   | 0.00     | 1       |
| 6     | 0.80      | 0.90   | 0.85     | 149     |
| 7     | 0.94      | 0.89   | 0.91     | 149     |
| 14    | 0.00      | 0.00   | 0.00     | 3       |
| 17    | 0.00      | 0.00   | 0.00     | 7       |

**Accuracy**: 0.84 (593 samples)

**Macro Avg**:  
- Precision: 0.42  
- Recall: 0.43  
- F1-Score: 0.43  
- Support: 593  

**Weighted Avg**:  
- Precision: 0.83  
- Recall: 0.84  
- F1-Score: 0.83  
- Support: 593  

**Test Accuracy**: 0.8432

### Overall End-to-End Test Results for 3 categories

**ECE Weighted Avg**:  
- Precision: 0.91  
- Recall: 0.94  
- F1-Score: 0.92  
- Support: 251  

**Pysch Weighted Avg**:  
- Precision: 0.83  
- Recall: 0.85  
- F1-Score: 0.84  
- Support: 303  

**BioChem Weighted Avg**:  
- Precision: 0.83  
- Recall: 0.84  
- F1-Score: 0.83  
- Support: 593  

### Conclusion
- The macro avg for level 2 classifiers should be ignored, because these come from misclassification from the main classifier
- Overall ECE performed quite well relative to the paper, but Psychology and Biochemistry were a bit lower
- For pyschology, the recall for class 7 was quite low, which is `Depression` mostly because its complicated / diverse nature. Class 17 which belongs to "attention" was also low.
- for biochem 0, 3, and 6 scored low, which are `Molecular biology`, `Immunology`, and `Polymerase chain reaction` respectively, not sure why will have to dig deeper.

### Next Steps
- See the effect of keywords on level 2 models
- do hyperparameter tuning for completness sake but I do not expect much improvement
- Try BERT like model especially for level 2 models

## LLMs