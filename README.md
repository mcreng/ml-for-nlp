## HKUST COMP4901K/MATH4824B Project Repository

This repository stores all project codes from said course, titled 'Machine Learning with Natural Language Processing'

### Project 1

Project 1 is about Naive Bayes. It requires students to compute Naive Bayes prior, likelihood and posterior probabilities. Used codes are attached as well. See `Proj1/proj1_q.pdf` for more information.

### Project 2

Project 2 is about Sentimental Analysis of text. Students are required to train their best models and compete with one another in [kaggle](https://www.kaggle.com/c/ml4nlp-sentiment-analysis). In this project, different methods are attempted.

- `NaiveBayes.ipynb`: Code I adapted from sample codes, which used native numpy codes writing a Naive Bayes classifier.
- `NaiveBayes-sklearn.ipynb`: Naive Bayes classifier with sklearn library.
- `sklearn-MLP.ipynb`: Multilayer Perceptron classifier using sklearn library.
- `ModelEnsemble.ipynb`: An ensemble model combining Naive Bayes, Logistic Regression (notebook lost during development) and MLP.
- `Tensorflow.ipynb`: A quick notebook that I developed the night before assignment due date after realising I could have been using Tensorflow the whole time...

The model scores 7th out of 73 students.

### Project 3

Project 3 is about Language Model. Students are required to train a network which given the first n-1 words of a sentence, the probability distribution of the last word of the sentence. View the report in it for more details. In short, a model with [TCN](https://github.com/philipperemy/keras-tcn) and LSTM with Embedding residual connection is used and achieved a score well beyond bonus.