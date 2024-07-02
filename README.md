# Automated Text Summarization


The "Automated Text Summarization" repository hosts scripts for building and evaluating deep learning models that automate text summarization tasks. Using LSTM-based sequence-to-sequence models, it preprocesses and tokenizes datasets like CNN/Daily Mail, trains models for summarization, and evaluates them on test sets. The repository provides functionalities for training, testing, and evaluating model performance, supporting efficient text summarization tasks in natural language processing applications.

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
 
   
