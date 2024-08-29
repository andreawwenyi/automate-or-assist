import json
import pandas as pd
import re
import torch
from fastcoref import LingMessCoref

full = pd.read_csv("./data/maxqda_full_transcript.csv")
window_size = 20

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# --- Make sliding window dataframe --- #
paragraph_df = pd.DataFrame()
for defendant in full["defendant"].unique():
    print(defendant)
    d = full[full["defendant"] == defendant].copy()
    d["idx"] = d["paragraph_id"].apply(lambda x: x.split("-")[-1])

    paragraph_series = d["text"].shift(0)
    for w in range(1, window_size):
        paragraph_series = paragraph_series + " " + d["text"].shift(-1 * w)

    idx_series = d["idx"].shift(0)
    for w in range(1, window_size):
        idx_series = idx_series + "," + d["idx"].shift(-1 * w)

    x = pd.DataFrame(
        {
            "paragraph": paragraph_series,
            "paragraph_idx": idx_series,
            "target_sentence": d["text"].shift(-1 * w),  # last sentence
        }
    )
    x["defendant"] = defendant
    x = x[x["paragraph"].notnull()]
    x["paragraph"] = x["paragraph"].apply(
        lambda p: re.sub("\s?[\d]+\s", " ", p).strip()
    )
    x["target_sentence"] = x["target_sentence"].apply(
        lambda p: re.sub("\s?[\d]+\s", " ", p).strip()
    )
    x["target_sentence_start_index"] = x.apply(
        lambda row: row["paragraph"].rindex(row["target_sentence"]), axis=1
    )
    paragraph_df = pd.concat((paragraph_df, x))


# --- Function to identify the coreference in the target sentence (i.e. the last sentence in a 20-sentence paragraph) --- #
def find_coref_in_target_sentence(row):
    coref_clusters = row["coref_preds"].get_clusters(as_strings=False)
    # for each cluster, if there is one start index >= target_sent_start_idx, print the cluster
    coreference_in_target_sent = list()
    for i, clust in enumerate(coref_clusters):
        for span in clust:
            if span is None:
                continue
            start_idx = span[0]
            if start_idx >= row["target_sentence_start_index"]:
                cluster = row["coref_preds"].get_clusters(as_strings=True)[i]
                coreference_in_target_sent.append(
                    list(set(cluster))
                )  # keep unique ones only
                break  # go to the next cluster directly
    return coreference_in_target_sent


# --- Run coreference resolution --- #
coref_model = LingMessCoref(device=device)
paragraphs = paragraph_df["paragraph"].to_list()

preds = coref_model.predict(texts=paragraphs)

paragraph_df["coref_preds"] = preds
paragraph_df["coref_in_target"] = paragraph_df.apply(
    find_coref_in_target_sentence, axis=1
)
paragraph_df = paragraph_df.drop("coref_preds", axis=1)

# --- Filter coreference resolution --- #
pronouns = [
    "you",
    "your",
    "yours",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "we",
    "us" "our",
    "i",
    "my",
    "me",
    "they",
    "their",
    "theirs",
    "them",
    "it",
    "its",
    "it's",
    "that",
    "this",
    "herself",
    "himself",
    "myself",
    "youself",
    "themselves",
    "yourselves",
]

nicknames = json.load(open("./data/nicknames.json", "r"))


def is_about_defendant(row):
    defendant = row["defendant"]
    names = defendant.split(" ")
    possible_mentions = [
        names[0].lower(),  # first name
        names[-1].lower(),  # last name
        names[0].lower()
        + " "
        + names[-1].lower(),  # first name + last name (skip middle name)
        f"ms. {names[-1].lower()}",
        f"miss {names[-1].lower()}",
        f"mrs. {names[-1].lower()}",
        f"ms. {names[0].lower()}",
        "this woman",
        "this woman over here",
        defendant.lower(),  # full name
        "the defendant",
        "this defendant",
        "the defendant in this case",
        "defendant",
        "my client",
    ]
    possible_mentions = possible_mentions + nicknames.get(defendant, [])

    for cluster in row["coref_in_target"]:
        # take the unique set of nouns
        unique_nouns = set(cluster)
        except_pronouns = set(cluster).difference(pronouns)
        # if there's anything other than pronouns
        if except_pronouns:
            # check if any unique noun mention the defendant
            mentions = except_pronouns.intersection(possible_mentions)
            if mentions:
                return True
            else:  # go to the next cluster
                continue
        else:  # if nothing other than pronouns
            if unique_nouns.intersection(["she", "her"]):
                return True
            else:
                continue
    return False


paragraph_df["target_mentions_defendant"] = paragraph_df.apply(
    is_about_defendant, axis=1
)
paragraph_df["paragraph_id"] = paragraph_df.apply(
    lambda row: row["defendant"].replace(" ", "_")
    + "-"
    + row["paragraph_idx"].split(",")[-1],
    axis=1,
)
paragraph_df.to_csv("./data/coref.csv", index=False)
