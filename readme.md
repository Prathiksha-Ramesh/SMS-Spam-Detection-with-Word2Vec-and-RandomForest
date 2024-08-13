# SMS Spam Detection with Word2Vec and RandomForest

This repository contains the source code and resources for an SMS Spam Detection project that utilizes Word2Vec embeddings and a RandomForest classifier to classify SMS messages as spam or ham (not spam).

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Word2Vec Embeddings](#word2vec-embeddings)
  - [Model Training and Evaluation](#model-training-and-evaluation)
- [Model Evaluation](#model-evaluation)
- [License](#license)
- [Contact](#contact)

## Project Overview

The SMS Spam Detection project leverages Word2Vec word embeddings and a RandomForest classifier to detect spam messages. Word2Vec is used to convert text data into numerical vectors that capture semantic meanings, which are then fed into a RandomForest model for classification.

## Dataset

The dataset used in this project is the **SMSSpamCollection**, which contains SMS messages labeled as spam or ham. The dataset is preprocessed to clean the text, apply lemmatization, and convert text into Word2Vec embeddings.

## Project Structure

- **notebook.ipynb**: The Jupyter notebook containing the complete code for data preprocessing, Word2Vec embedding generation, model training, evaluation, and visualization.
- **SMSSpamCollection**: The dataset file containing labeled SMS messages.
- **LICENSE**: The Apache License 2.0 file that governs the use and distribution of this project's code.
- **requirements.txt**: A file listing all the Python libraries and dependencies required to run the project, including Gensim, NLTK, Pandas, Scikit-learn, and Numpy.
- **.gitignore**: A file specifying which files or directories should be ignored by Git.

## Installation

To run the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-repository-name.git
```

2. Navigate to the project directory:

``` bash 
cd your-repository-name
``` 

3. Create a virtual environment (optional but recommended):

``` bash 
python3 -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
``` 
4. Install the required dependencies:

``` bash
pip install -r requirements.txt
``` 

5. Run the Jupyter notebook:
``` bash 
jupyter notebook notebook.ipynb
```
## Usage

### Word2Vec Embeddings

1. Text Preprocessing:

The text data is cleaned by removing non-alphabetic characters, converting text to lowercase, tokenizing the text, and applying lemmatization using `WordNetLemmatizer`.

2. Word2Vec Model:

- The `word2vec-google-news-300` pretrained model is loaded using Gensim to leverage word embeddings for the text data.
- A custom Word2Vec model is also trained on the tokenized words from the corpus.
- Each SMS message is converted into an average Word2Vec embedding vector by averaging the vectors of the words contained in the message.


## Model Training and Evaluation

1. RandomForest Classifier:

- The extracted Word2Vec features are used to train a RandomForest classifier.
- The model is trained on the training data and evaluated on the test data.

2. Model Evaluation:

The model is evaluated on the test set using the following metrics:

- Accuracy: The ratio of correctly predicted instances to the total instances.
- Classification Report: Provides precision, recall, and F1-score for both spam and ham classes.
- Confusion Matrix: Visualizes the performance of the classification model.

## License
This project is licensed under the Apache License 2.0. See the `LICENSE` file for more details.

## Contact
For any inquiries or contributions, feel free to reach out or submit an issue or pull request on GitHub.

