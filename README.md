
# NLP- and Layout-based Title Detection

A pipeline for classifying whether a text-box in a PDF is a title for structured document processing and downstream tasks such as efficient automated summarisation, or digital archiving.

The pipeline achieved **98.50% accuracy** and a combined **F-1 score of 98.05%** on a test dataset using a Random Forest classifier.

## Features
- `Contains Finite Verb`: Boolean, whether the text contains a regular finite verb common in long-form description but rare in a concise title. (e.g. “The Company **has** two equity participation plans”, “costs **are** subject to audit”) versus participles or infinitive verb (e.g. “**Consolidated** Statements of Comprehensive Income”, “Country by country **reporting**”)
- `Contains Full Stop`: Boolean, statistically-determined to be unlikely in a Title
- `Contains Pronoun`: Boolean, pronouns are unlikely in titles
- `All Numerical`: Boolean, captures purely numerical elements
- `All Upper Case`: Boolean, high correlation with titles
- `Ends in Colon`: Boolean, titles often introduce sections
- `Text String Length`: Integer, titles are usually shorter
- `Percentage Upper Case`: Float, higher in titles
- `Height-to-Width Ratio`: Float, titles are wider than they are tall
- `Area of Text Box`: Integer, titles are generally smaller in area
- `Centre X and Y Coordinates`: Positional indicators
- `Percentage of Nouns`: Float, titles contain more noun phrases
- `Is Bold`, `Is Italic`, `Is Underlined`: Boolean font style indicators
- `Left`, `Right`, `Top`, `Bottom`: PDF coordinates of the textbox
- `Text`: Raw text string

All continuous variables were standardized with Z-score normalization. The pipeline extracts linguistic features from the raw text using the NLTK library's on part-of-speech tagging.

## Sample Statistical Analysis
All of the engineered boolean features show high class-conditional probability for the Titles class, especially 'All upper case' (0.97) and 'Contains Finite Verb' (0.95). 'All Upper Case' and 'All Numerical' also shows high specificity as well as sensitivity, since they exclude non-Titles with a probability of 0.76 in both cases.

### Class-conditional probabilities for boolean features
![image](https://github.com/user-attachments/assets/dc052bd5-a1cd-4aef-87dc-a33241ae850c)

Many of the integer and continuous variables demonstrate good potential for linear separability of classes, for example `Text String Length` at the value 100 or `Area` at the value 10000.

### Probability densities of each class for continuous variables - Titles in Red, Non-Titles in Blue
![image](https://github.com/user-attachments/assets/0c9513e1-74a2-46d2-b88b-c4e7a31994c9)


Others features show considerable overlap such as `Top` and `Centre Y Coordinate`, but these may be useful features in combination with others, as can be seen on the following plots which combine features to show the top-left corner of each text-box and also the bottom-right corner.

### Positions of textboxes on a PDF page for both Titles (Red) and Non-titles (Blue)
![image](https://github.com/user-attachments/assets/dccba590-bc78-4c8b-b4f2-1e77659c1160)

All analysis was conducted on a training dataset only.

## Dimensionality Reduction

Many of the features may be colinear and therefore hinder training with added redundancy, so it was determined that these could be compressed from 20 features to an optimal 7 features with minimal performance loss. 
This optimum was found after performing principal component analysis with a varying number of components, training each with a decision tree classifier and tested on the validation dataset partition. Performance was found to be roughly equal for 7 or more components (97.0% accuracy at 7 components versus 98.3% at 20 components)

## Model Performance
A Random Forest classifier was found to be the best performing out of a selection of different architectures.

| Class       | Precision | Recall | F1-Score | Support |
|------------|----------|-------|---------|---------|
| Non-Titles | 98.4%    | 99.7%  | 99.0%   | 1174    |
| Titles     | 99.0%    | 95.3%  | 97.1%   | 405     |
| **Overall Accuracy** | | | **98.5%** | 1579    |

## Model Architecture Comparison

| Model                   | F1-Score | Precision | Recall | Accuracy |
|------------------------|---------|-----------|-------|----------|
| **Random Forest**       | **97.1%** | 99.0%     | 95.3%  | 98.5%    |
| Multilayer Perceptron   | 90.9%   | 90.7%     | 91.1%  | 95.3%    |
| Logistic Regression     | 94.7%   | 96.4%     | 93.1%  | 97.3%    |
| Naïve Bayes (Gaussian)   | 73.2%   | 58.7%     | 97.3%  | 81.8%    |
| Decision Tree           | 93.6%   | 94.7%     | 92.6%  | 96.8%    |


## Model Limitations

- Training data was composed mostly of financial compliance documents which increases the likelihood of domain bias outside of this format
- The dataset is roughly imbalanced (non-Titles outnumber Titles by 2.89x) which can lower sensitivity to Title classes - this can be improved by random undersampling or SMOTE based oversampling
- Features based on PDF text-boxes (i.e. coordinates, area) are not likely to generalise to other document formats such as slide decks or posters
- Linguistic features (part-of-speech tags, capitalisation, punctuation) are based on those in the English language and may not present comparable results in other languages


## Requirements

```bash
pandas
scikit-learn
nltk
matplotlib
seaborn
scipy
joblib
```
