# Credit Card Fraud Detection using Random Forest and ANN

This project implements two machine learning models, Random Forest and Artificial Neural Network (ANN), to detect fraudulent credit card transactions. The dataset consists of transactions labeled as either legitimate (class 0) or fraudulent (class 1), with a strong imbalance between the two classes. The goal of the project is to classify transactions effectively while addressing the challenges posed by this class imbalance.

## Project Workflow

1. **Data Preprocessing**: 
   - Load the credit card fraud detection dataset.
   - Split the dataset into training and test sets, ensuring the test set maintains the class distribution.
   
2. **Model Training**:
   - **Random Forest Classifier**:
     - A Random Forest model is trained on the preprocessed dataset.
   - **Artificial Neural Network (ANN)**:
     - A feedforward neural network (ANN) is built with multiple dense layers and trained using binary cross-entropy loss.

3. **Model Evaluation**:
   - Both models are evaluated using accuracy, precision, recall, F1-score, and confusion matrix.
   - Special attention is paid to the model's ability to correctly detect fraudulent transactions (class 1).

## Model Results

### 1. Random Forest Classifier
The Random Forest model displayed strong performance in classifying fraudulent transactions, even with the imbalanced dataset.

- **Classification Report**:
  ```
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     85296
           1       0.95      0.74      0.83       147

    accuracy                           1.00     85443
   macro avg       0.97      0.87      0.92     85443
weighted avg       1.00      1.00      1.00     85443```

- **Classification Report**:
  ```
  [[85290     6]
   [   38   109]]
  ```

- **Interpretation**:
  - Precision (class 1): 0.95, indicating that 95% of the transactions predicted as fraudulent are actually fraudulent.
  - Recall (class 1): 0.74, meaning 74% of actual fraudulent transactions were correctly identified.
  - F1-Score: The F1-score for fraudulent transactions is 0.83, reflecting the balance between precision and recall.
  - Overall Accuracy: The model achieved an accuracy of 99.95%, showcasing excellent performance.

### 2. Artificial Neural Network (ANN)
The ANN model struggled with the class imbalance, failing to detect any fraudulent transactions during evaluation.

- Classification Report:
  ```
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     85296
           1       0.00      0.00      0.00       147

    accuracy                           1.00     85443
   macro avg       0.50      0.50      0.50     85443
weighted avg       1.00      1.00      1.00     85443```

- **Confusion Matrix**:
  ```
  [[85296     0]
   [  147     0]]```

- **Interpretation**:
- Precision (class 1): 0.00, as the model did not identify any fraudulent transactions.
- Recall (class 1): 0.00, indicating that none of the actual fraudulent transactions were correctly classified.
- The ANN model overfits to the majority class (class 0) and completely misses the minority class (class 1), suggesting further adjustments like resampling or adjusting class weights are necessary.

### 2. Conclusion

- Random Forest significantly outperformed the ANN in this task, achieving high precision and recall for detecting fraud.
- The ANN model failed to generalize to the minority class (fraudulent transactions), possibly due to class imbalance. Future improvements could include rebalancing the dataset or adjusting the model’s training procedure.



### 3. Important Note

The models in this project were trained and evaluated **without explicitly addressing the data imbalance**. This decision was made deliberately to assess and compare how well each model generalizes to an imbalanced dataset where fraudulent transactions (class 1) are vastly outnumbered by legitimate ones (class 0). 
Random Forest managed to perform reasonably well despite the imbalance, while the **ANN model** struggled to identify fraudulent transactions correctly.

### Handling Data Imbalance
To improve model performance on imbalanced datasets like this one, the following techniques can be applied:

- Resampling the Dataset:
  - Oversampling the minority class (fraud) or undersampling the majority class (legitimate) to create a more balanced distribution.
  
- Adjusting Class Weights:
  - Assigning higher weights to the minority class during model training helps the model pay more attention to the underrepresented class.
