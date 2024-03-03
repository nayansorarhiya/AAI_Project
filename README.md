# Phase3
## Bias in AI
- As AI-based applications increasingly become a part of everyday life, the analysis and mitigation of biases -often inadvertently introduced through training data -have emerged as critical concerns.
- In this part of the project, our challenge is to ensure that our AI system is not only technically procient but also ethically sound. We are required to analyze our AI for two of the following categories: age and gender. For example, using suitably annotated testing data, investigate
whether our system's performance across the four classes remains consistent across dierent genders. Experiment with methods such as re-balancing or augmenting our training dataset to address any identified biases. Our evaluation should encompass the network's performance on both the complete
dataset and the subsets representing the chosen biased attributes.

## Evaluation: K-fold Cross-validation
- Building upon the basic train/test split methodology from Part II, this phase requires me to employ k-fold cross-validation to enhance the robustness and reliability of our evaluation, especially across different classes.
- We are instructed to perform a 10-fold cross-validation (with random shuffling) on our AI model (the final model from Part II). This method involves dividing our dataset into 10 equal parts, using each part once as a test set while the remaining nine parts serve as the training set. This process is repeated
10 times, each time with a different part as the test set. This technique helps in assessing the model's performance more comprehensively and reduces the bias that can occur in a single train/test split.
- Utilize the functionalities provided by scikit-learn or skorch for implementing the k-fold cross-validation. It is important to avoid a manual, static split of the dataset for this process.
- Document the results of our 10-fold cross-validation in the project report. This includes the performance metrics (accuracy, precision, recall, F1-score) for each fold, as well as the average across all 10 folds.

## Detecting and Mitigating Bias
- This critical phase of our project involves analyzing our AI model for potential biases and implementing strategies to mitigate them. The process is divided into three main steps: data collection, analysis for bias, and retraining the system if needed. Start from the model we developed and saved in Part II of the project for the initial bias evaluation.
