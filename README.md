# GPT from Scratch

A character-level GPT (decoder-only Transformer) built from scratch in PyTorch, following Andrej Karpathy's "Neural Networks: Zero to Hero" lecture on building GPT. I wrote the notebook while following the lecture and added my own notes on tensor shapes and attention.

## What's in the notebook
- Character-level tokenizer and train/validation split on the Tiny Shakespeare dataset
- Bigram language model as a baseline
- The maths behind self-attention: from averaging previous tokens, to a lower-triangular matrix, to masked softmax
- Single-head self-attention (key, query, value), then LayerNorm
- Full model: token and position embeddings, multi-head attention, feed-forward layers, residual connections, LayerNorm and dropout
- Training loop with AdamW and train/validation loss evaluation, plus text generation

## Model settings
Batch size 64, context length 256, embedding size 384, 6 heads, 6 layers, dropout 0.2, learning rate 3e-4, 5000 iterations.

## How to run
Open `GPT_From_Scratch.ipynb` in Google Colab, switch to a GPU runtime and run all cells. The dataset downloads automatically.

## Credit
Based on Andrej Karpathy's Zero to Hero series.

## Next steps
Train the same model on a custom dataset of my own.
