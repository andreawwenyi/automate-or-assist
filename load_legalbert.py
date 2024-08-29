from transformers import AutoTokenizer
from transformers import pipeline


class legalbert:
    def __init__(self, model_path):
        # pipeline
        self.pipe = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased"),
            device=0,
        )
        self.tokenizer_kwargs = {"truncation": True, "max_length": 512}

    def score(self, corpus):
        pred = self.pipe(corpus, **self.tokenizer_kwargs)
        labels = []
        scores = []
        for p in pred:
            if p["label"] == "LABEL_1":
                labels.append(1)
                scores.append(p["score"])
            else:
                labels.append(0)
                scores.append(1 - p["score"])
        return scores
