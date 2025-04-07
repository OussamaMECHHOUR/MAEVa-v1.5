import argparse
import os

import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Embedding extraction ===
def get_sentence_embedding(sentence, tokenizer, model):
    inputs = tokenizer(
        sentence, max_length=512, truncation=True, padding="max_length", return_tensors="pt"
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        outputs = model.bert(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
    return cls_embedding.cpu().numpy()


def evaluate_similarity(args):
    print("Loading model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = BertForMaskedLM.from_pretrained(args.model_path).to(device)
    model.eval()

    print("Loading source and candidate descriptions...")
    with open(args.source_file, "r", encoding="utf-8") as f:
        sources = [line.strip() for line in f]
    with open(args.candidate_file, "r", encoding="utf-8") as f:
        candidates = [line.strip() for line in f]

    print("Generating embeddings...")
    source_embeddings = [get_sentence_embedding(sent, tokenizer, model) for sent in tqdm(sources)]
    candidate_embeddings = [get_sentence_embedding(sent, tokenizer, model) for sent in tqdm(candidates)]

    print("Computing similarities...")
    results = []
    for source, source_emb in zip(sources, source_embeddings):
        similarities = [
            (candidate, cosine_similarity(source_emb, candidate_emb)[0][0])
            for candidate, candidate_emb in zip(candidates, candidate_embeddings)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_10 = similarities[:10]
        results.append({
            "source_phrase": source,
            "top_10_candidates": [cand[0] for cand in top_10],
            "top_10_scores": [cand[1] for cand in top_10]
        })

    os.makedirs(os.path.dirname(args.output_results), exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_excel(args.output_results, index=False)
    print(f"Similarity results saved to: {args.output_results}")

    print("Validating with ground-truth correspondences...")
    validation_df = pd.read_excel(args.validation_file)
    validation_results = []
    top_1_correct = top_3_correct = top_5_correct = top_10_correct = 0

    for _, row in validation_df.iterrows():
        source = row["Variable source"]
        expected = row["Variable correspondante"]
        top_candidates = next((r["top_10_candidates"] for r in results if r["source_phrase"] == source), [])

        is_top_1 = expected in top_candidates[:1]
        is_top_3 = expected in top_candidates[:3]
        is_top_5 = expected in top_candidates[:5]
        is_top_10 = expected in top_candidates[:10]

        if is_top_1: top_1_correct += 1
        if is_top_3: top_3_correct += 1
        if is_top_5: top_5_correct += 1
        if is_top_10: top_10_correct += 1

        validation_results.append({
            "source_phrase": source,
            "expected_candidate": expected,
            "is_in_top_1": is_top_1,
            "is_in_top_3": is_top_3,
            "is_in_top_5": is_top_5,
            "is_in_top_10": is_top_10,
        })

    total = len(sources)
    summary = {
        "Validation Type": ["Top 1", "Top 3", "Top 5", "Top 10"],
        "Correct Matches": [top_1_correct, top_3_correct, top_5_correct, top_10_correct],
        "Total Phrases": [total] * 4,
        "CMP (%)": [
            100 * top_1_correct / total,
            100 * top_3_correct / total,
            100 * top_5_correct / total,
            100 * top_10_correct / total,
        ]
    }

    df_validation = pd.DataFrame(validation_results)
    df_summary = pd.DataFrame(summary)

    os.makedirs(os.path.dirname(args.output_validation), exist_ok=True)
    with pd.ExcelWriter(args.output_validation) as writer:
        df_validation.to_excel(writer, sheet_name="Validation Details", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Validation results saved to: {args.output_validation}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code to reproduce our best results for matching variable descriptions using our best pretrained AgriBERT model.")

    parser.add_argument("--model_path", default="scripts/Further pretraining/saved models/our_pretrained_AgriBERT", help="Path to the weights and tokenizer of our best pretrained AgriBERT model")
    parser.add_argument("--source_file", default="datasets/benchmarks/descriptions_sources.txt", help="Path to source descriptions file (.txt)")
    parser.add_argument("--candidate_file", default="datasets/benchmarks/descriptions_candidates.txt", help="Path to candidate descriptions file (.txt)")
    parser.add_argument("--validation_file", default="datasets/benchmarks/Correspondances.xlsx", help="Path to Excel file with ground-truth correspondences")
    parser.add_argument("--output_results", default="outputs/matching_variables/our_pretrained_AgriBERT/agriBERT_results_evaluation.xlsx", help="Path to save top-k matching results (.xlsx)")
    parser.add_argument("--output_validation", default="outputs/matching_variables/our_pretrained_AgriBERT/agriBERT_validation_evaluation.xlsx", help="Path to save validation results (.xlsx)")

    args = parser.parse_args()
    evaluate_similarity(args)
