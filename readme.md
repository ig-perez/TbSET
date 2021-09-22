# Notes
- `local.multivac` is just a storage folder with a random name.

# General


1. Load dataset for training
2. Tokenize the datasets for training
    2.1. Create a vocabulary for our dataset using the Wordpiece algorithm
    2.2. Use the vocabulary to build a custom tokenizer based on BERT tokenization
    2.3. Create the tokenizers (objects) for both languages
3. Prepare input data for training (one line)
    3.1. Cache the datasets
    3.2. Shuffle the datasets
    3.3. Create batches
    3.4. Tokenize the datasets
    3.5. Prefetch the datasets

4. Train
    4.1. Instantiate the Transformer
    4.2. Set up the checkpoint manager
    4.3. Obtain batched training dataset
    4.4. Run the training step for each epoch to process all training dataset