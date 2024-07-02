# Automated Text Summarization

This repository contains code for training and evaluating models for automated text summarization using deep learning techniques. It includes preprocessing steps, model training, evaluation on test data, and utilities for tokenization and sequence padding.

## Features

- **Data Preprocessing**: Includes text cleaning, tokenization, and padding of sequences.
- **Model Architecture**: LSTM-based encoder-decoder model with attention mechanism for text summarization.
- **Training and Evaluation**: Scripts for training models, evaluating performance metrics, and saving tokenizers for reproducibility.
- **Utilities**: Tokenizer management and data generator functions for efficient model training.

## Requirements

- TensorFlow
- Keras
- NumPy
- Pandas
- NLTK
- tqdm
- rouge-score

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Automated-Text-Summarization.git
   cd Automated-Text-Summarization
  ```

2.Install dependencies:
  ```bash
  pip install -r requirements.txt
   ```
3. Run the scripts:
  ```bash
  python train_model.py
  ```
## Usage

1. Prepare your dataset in CSV format with 'article' and 'highlights' columns.
2. Adjust model parameters and hyperparameters in the scripts as needed.
3. Run scripts for training, evaluation, and inference.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please contact shivamdave172003@gmail.com .
 
   
