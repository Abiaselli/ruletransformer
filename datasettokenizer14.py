import os
import json
import time
import concurrent.futures
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers

def process_json_file(path, lowercase=True):
    """ Read and process a JSON file, returning text samples. """
    texts = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, dict):
                texts = [data.get("text", str(data))]
            elif isinstance(data, list):
                texts = [str(item) for item in data]
            else:
                texts = [str(data)]

        # Lowercase if needed
        if lowercase:
            texts = [text.lower() for text in texts]

        return texts
    except Exception as e:
        print(f"[ERROR] Could not read {path}: {e}")
        return []

def iterate_texts_from_dataset(dataset_dir, lowercase=True, batch_size=5000):
    """
    Yields batches of text from JSON files using parallel file reading.
    """
    file_list = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.lower().endswith('.json')]
    total_files = len(file_list)

    print(f"\n[INFO] Found {total_files} JSON files. Processing in batches of {batch_size} samples.")
    
    # Use ThreadPoolExecutor for parallel file reading
    count = 0
    buffer = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(lambda f: process_json_file(f, lowercase), file_list)

        for file_idx, texts in enumerate(results, start=1):
            if not texts:
                continue

            for text in texts:
                buffer.append(text)
                count += 1

                # Yield a batch when buffer reaches batch_size
                if len(buffer) >= batch_size:
                    print(f"  --> Yielding batch of {len(buffer)} samples. Total processed: {count}")
                    yield buffer
                    buffer = []

            print(f"[INFO] Finished file {file_idx}/{total_files}")

    # Yield remaining texts
    if buffer:
        print(f"  --> Yielding final batch of {len(buffer)} samples.")
        yield buffer

def build_tokenizer_from_dataset(dataset_dir, output_path, vocab_size=200000, use_bytelevel=True, lowercase=True, batch_size=5000, frequency_cutoff=50):
    """ 
    Build and train a tokenizer incrementally from a dataset of JSON files using parallel file reading.
    """
    print("\n[INFO] Initializing tokenizer...")
    if use_bytelevel:
        tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Punctuation(),
            pre_tokenizers.Split(r"(\d)", behavior='isolated')
        ])
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, 
            min_frequency=frequency_cutoff, 
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
            continuing_subword_prefix="##",
            initial_alphabet=list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        )
    else:
        tokenizer = Tokenizer(models.WordLevel(unk_token="<UNK>"))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Punctuation(),
            pre_tokenizers.Split(r"(\d)", behavior='isolated')
        ])
        tokenizer.decoder = decoders.WordLevel()
        trainer = trainers.WordLevelTrainer(
            vocab_size=vocab_size, 
            min_frequency=frequency_cutoff, 
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
            initial_alphabet=list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        )

    print("\n[INFO] Starting tokenizer training with batch processing...\n")
    
    start_time = time.time()
    
    for batch_num, batch in enumerate(iterate_texts_from_dataset(dataset_dir, lowercase=lowercase, batch_size=batch_size), start=1):
        if batch_num % 10 == 0:  # Print updates every 10 batches
            print(f"[TRAINING] Processing batch {batch_num} of {batch_size} samples...")

        tokenizer.train_from_iterator(batch, trainer=trainer)

    print("\n[INFO] Tokenizer training completed in {:.2f} seconds.".format(time.time() - start_time))

    tokenizer.save(output_path)
    print(f"\n[SUCCESS] Tokenizer saved to {output_path}")

    return tokenizer

# Example usage:
dataset_dir = r"C:\Users\abias\alldatareasoning"
output_path = "tokenizer_from_dataset_bytelevel_new_2.json"
tokenizer = build_tokenizer_from_dataset(dataset_dir, output_path, vocab_size=100000, use_bytelevel=True, batch_size=10000, frequency_cutoff=5)
