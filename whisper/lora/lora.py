import argparse
import json
import math
import time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import utils as lora_utils

from mlx.utils import tree_flatten, tree_unflatten
from models.lora import LoRALinear

from models.decoding import DecodingOptions, Inference

# Audio Feature Extractor
from models.audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)


# Configure typealias for batched inputs
from collections import namedtuple

BatchInput = namedtuple("BatchInput", "audio sentence")
issubclass(BatchInput, tuple)


def build_parser():
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--model",
        default="mlx_whisper_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/",
        help="Directory with {train, valid, test}.jsonl files",
    )
    # Generation args
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="The maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=0.8, help="The sampling temperature"
    )

    # Training args
    parser.add_argument(
        "--train",
        action="store_true",
        help="Do training",
    )

    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="mozilla-foundation/common_voice_16_0",
        help="HuggingFace dataset to use for training",
    )

    parser.add_argument(
        "--hf-dataset-lang",
        type=str,
        default="te",
        help="HuggingFace dataset's language to use for training",
    )

    parser.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of layers to fine-tune",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Minibatch size.")
    parser.add_argument(
        "--iters", type=int, default=1000, help="Iterations to train for."
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=25,
        help="Number of validation batches, -1 uses the entire validation set.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5, help="Adam learning rate."
    )
    parser.add_argument(
        "--steps-per-report",
        type=int,
        default=10,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        default=200,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--resume-adapter-file",
        type=str,
        default=None,
        help="Load path to resume training with the given adapter weights.",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        default="adapters.npz",
        help="Save/load path for the trained adapter weights.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save the model every N iterations.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set after training",
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        default=500,
        help="Number of test set batches, -1 uses the entire test set.",
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    return parser


def load(args):
    if args.hf_dataset is not None:
        hf_dataset = args.hf_dataset
        print(f"Using hf dataset {hf_dataset}...")
    else:
        print("Falling back to default mozilla common voice ")
        hf_dataset = "mozilla-foundation/common_voice_16_0"

    if args.hf_dataset_lang is not None:
        hf_dataset_lang = args.hf_dataset_lang
        print(f"Using dataset lang {hf_dataset_lang}...")
    else:
        print("Falling back to telugu lang")
        hf_dataset_lang = "te"

    print(f"Loading dataset {hf_dataset}, {hf_dataset_lang} from hugging face")
    # todo: whisper-lora: select only necessary columns audio, sentence using dataset.select_columns
    
    """
    Load a CSV dataset and split it into training, validation, and test sets.
    
    The CSV is expected to have the following columns:
        - filename: the path to the audio file (e.g., "Add a medication for sinemet.m4a")
        - reference: the ground truth transcription.
        - whisper: transcription from Whisper (optional).
        - apple_speech: transcription from Apple Speech (optional).
    
    This function renames 'filename' to 'path' and 'reference' to 'sentence'
    so that it is compatible with the rest of the code.
    
    Parameters:
        csv_path (str): Path to the CSV file.
        seed (int): Random seed for reproducible splits.
        test_size (float): Fraction of data to reserve for testing.
        valid_size (float): Fraction of data (from the remaining data) to reserve for validation.
        
    Returns:
        tuple: (train, valid, test) where each is a dict with keys 'path' and 'sentence' 
            and values as lists.
    """
    test_size=0.1
    valid_size=0.1
    seed = 42
    # Read the CSV file into a DataFrame
    df = pd.read_csv("/Users/andreas/Desktop/Whisper+PD/merged_si.csv")
    
    # Rename columns to match expected names in your training code
    df = df.rename(columns={"filename": "path", "reference": "sentence"})
    
    # Shuffle the DataFrame using the provided seed for reproducibility
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # First, split off the test set
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    
    # Next, split the remaining data into training and validation sets.
    # Calculate validation fraction relative to the remaining train_val data.
    valid_relative_size = valid_size / (1 - test_size)
    train_df, valid_df = train_test_split(train_val_df, test_size=valid_relative_size, random_state=seed)
    
    # Convert the DataFrames to dictionaries with lists so that they are compatible
    # with your iterate_batches function (which accesses dset["path"] and dset["sentence"])
    train = train_df.to_dict(orient="list")
    valid = valid_df.to_dict(orient="list")
    test = test_df.to_dict(orient="list")
    
    if args.train and len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if args.train and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if args.test and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train, valid, test


def iterate_batches(dset, tokenizer, batch_size, train=False):
    # Instead of computing target_frames dynamically, use the fixed expected frame count.
    target_frames = N_FRAMES  # use the constant expected by the model

    while True:
        # Shuffle indices if training
        indices = np.arange(len(dset["path"]))
        if train:
            indices = np.random.permutation(indices)

        # Collect batches from the dataset
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            # Get file paths for the current batch
            paths = [dset["path"][indices[i + j]] for j in range(batch_size)]
            # Process each audio sample and ensure it has the expected number of frames.
            batch_audio = [
                pad_or_trim(
                    log_mel_spectrogram(path, padding=N_SAMPLES),
                    target_frames,  # now always N_FRAMES
                    axis=-2,
                ).astype(mx.float32)
                for path in paths
            ]
            # Encode sentences using the tokenizer
            batch_sentence = [
                tokenizer.encode(dset["sentence"][indices[i + j]])
                for j in range(batch_size)
            ]
            assert len(batch_sentence) == len(batch_audio), "unequal batches of text & audio lengths"

            # Compute shapes for each processed audio sample and token list lengths
            shapes_audio = [x.shape for x in batch_audio]
            lengths_sentence = [len(x) for x in batch_sentence]

            # Warn if any sequence exceeds typical limits
            if max(lengths_sentence) > 2048 or max(shapes_audio)[0] > target_frames:
                print(
                    "[WARNING] Some sequences are longer than 2048 tokens and/or longer than the expected frame count. "
                    "Consider pre-splitting your data to save memory."
                )

            # Pad text sequences to the maximum sentence length in the batch
            batch_arr_sentence = np.zeros((batch_size, max(lengths_sentence)), np.int32)
            # Pad audio sequences to the maximum shape in the batch.
            batch_arr_audio = np.zeros((batch_size,) + max(shapes_audio), np.float32)

            # Fill the padded arrays with the actual sentence data
            for j in range(batch_size):
                batch_arr_sentence[j, :lengths_sentence[j]] = batch_sentence[j]
            batch_sentence = mx.array(batch_arr_sentence)

            # Fill the padded arrays with the actual audio data
            for j in range(batch_size):
                batch_arr_audio[j, :shapes_audio[j][0], :shapes_audio[j][1]] = batch_audio[j]
            batch_audio = mx.array(batch_arr_audio)

            # Yield the batch as a named tuple
            yield BatchInput(audio=batch_audio, sentence=batch_sentence)

        if not train:
            break


def evaluate(model, dataset, loss, tokenizer, batch_size, num_batches):
    all_losses = []
    ntokens = 0
    for it, batch in zip(
        range(num_batches),
        iterate_batches(dataset, tokenizer, batch_size),
    ):
        losses, toks = loss(model, *batch)
        all_losses.append((losses * toks).item())
        ntokens += toks

    return np.sum(all_losses) / ntokens


def loss(model, mels, tokens):
    # Run model on inputs to get logits
    logits = model(mels, tokens)  # logits shape: (batch, seq_length, vocab_size)
    logits = logits.astype(mx.float32)

    # Determine mask based on non-padding tokens.
    # Here we assume that the padding token ID is 0.
    # Adjust pad_token_id if needed.
    pad_token_id = 0
    # Create mask: 1 where token != pad_token_id, 0 otherwise.
    mask = (tokens != pad_token_id).astype(mx.float32)

    # Compute cross-entropy loss per token without reduction
    ce = nn.losses.cross_entropy(logits, tokens, reduction='none')
    # Multiply by the mask so that padded positions don't contribute.
    ce = ce * mask
    # Compute the total number of valid tokens
    ntoks = mask.sum()
    # Avoid division by zero
    ntoks = ntoks if ntoks > 0 else 1

    # Return average loss and token count
    return ce.sum() / ntoks, ntoks


def train(model, loss, tokenizer, args):
    # Load dataset
    print("Loading datasets")
    train_set, val_set, test_set = load(args)
    print(
        f"Loaded datasets with {len(train_set)} train, {len(val_set)} valid, {len(test_set)} test"
    )

    print("Training")
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    # Create value and grad function for loss
    loss_value_and_grad = nn.value_and_grad(model, loss)

    losses = []
    n_tokens = 0

    # Main training loop
    start = time.perf_counter()
    for it, batch in zip(
        range(args.iters),
        iterate_batches(train_set, tokenizer, args.batch_size, train=True),
    ):
        # Forward and backward pass
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)

        # Model update
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state, lvalue)

        # Record loss
        losses.append(lvalue.item())
        n_tokens += toks

        # Report training loss if needed
        if (it + 1) % args.steps_per_report == 0:
            train_loss = np.mean(losses)

            stop = time.perf_counter()
            print(
                f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                f"It/sec {args.steps_per_report / (stop - start):.3f}, "
            )
            losses = []
            n_tokens = 0
            start = time.perf_counter()

        # Report validation loss if needed
        if it == 0 or (it + 1) % args.steps_per_eval == 0:
            stop = time.perf_counter()
            val_loss = evaluate(
                model, val_set, loss, tokenizer, args.batch_size, args.val_batches
            )
            print(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val took {(time.perf_counter() - stop):.3f}s"
            )

            start = time.perf_counter()

        # Save adapter weights if needed
        if (it + 1) % args.save_every == 0:
            mx.savez(
                args.adapter_file, **dict(tree_flatten(model.trainable_parameters()))
            )
            print(f"Iter {it + 1}: Saved adapter weights to {args.adapter_file}.")


def freeze_and_lora(args, model):
    model.freeze()
    ## Apply LoRA to AudioEncoder
    print(f"Applying LoRA parameters to AudioEncoder...")
    for block in model.encoder.blocks[len(model.encoder.blocks) - args.lora_layers :]:
        block.attn.query = LoRALinear.from_linear(block.attn.query)
        block.attn.value = LoRALinear.from_linear(block.attn.value)
    print("Done applying Encoder LoRA Linear layers")
    enc_tot_params = (
        sum(v.size for _, v in tree_flatten(model.encoder.parameters())) / 10**6
    )
    print(f"Encoder: Total parameters {enc_tot_params:.3f}M")
    enc_tra_params = (
        sum(v.size for _, v in tree_flatten(model.encoder.trainable_parameters()))
        / 10**6
    )
    print(f"Encoder: Trainable parameters {enc_tra_params:.3f}M")
    ## Apply LoRA to TextDecoder
    print(f"Applying LoRA parameters to TextDecoder...")
    for block in model.decoder.blocks[len(model.decoder.blocks) - args.lora_layers :]:
        block.cross_attn.query = LoRALinear.from_linear(block.cross_attn.query)
        block.cross_attn.value = LoRALinear.from_linear(block.cross_attn.value)
    print("Done applying Decoder LoRA Linear layers")
    dec_tot_params = (
        sum(v.size for _, v in tree_flatten(model.decoder.parameters())) / 10**6
    )
    print(f"Decoder: Total parameters {dec_tot_params:.3f}M")
    dec_tra_params = (
        sum(v.size for _, v in tree_flatten(model.decoder.trainable_parameters()))
        / 10**6
    )
    print(f"Decoder: Trainable parameters {dec_tra_params:.3f}M")
    print("Finished adding LoRA params! :)")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    model, tokenizer, config = lora_utils.load(args.model)

    ### LoRA Telugu finetune Whisper using mozilla-foundation/common_voice_16_0
    # 1. Telugu is lowresource in common_16 with approx 300 rows
    # 2. Need to insert Adaptor layers in 2 places
    #     a. Once in AudioEncoder
    #     b. Once in TextDecoder
    # 3. Prep dataset
    #     a. Whisper encoder needs 16k HZ samples

    # # Freeze all layers & create LORA layers
    freeze_and_lora(args, model)

    # Resume training the given adapters.
    if args.resume_adapter_file is not None:
        print(f"Loading pretrained adapters from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)

    if args.train:
        # Train model
        train(model, loss, tokenizer, args)

        # Save adapter weights
        mx.savez(args.adapter_file, **dict(tree_flatten(model.trainable_parameters())))

    # Load the LoRA adapter weights which we assume should exist by this point
    if not Path(args.adapter_file).is_file():
        raise ValueError(
            f"Adapter file {args.adapter_file} missing. "
            "Use --train to learn and save the adapters.npz."
        )
    model.load_weights(args.adapter_file, strict=False)

    if args.test:
        print("Testing")
        model.eval()
        test_loss = evaluate(
            model,
            test_set,
            loss,
            tokenizer,
            args.batch_size,
            num_batches=args.test_batches,
        )
        test_ppl = math.exp(test_loss)

        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")
