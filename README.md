# automate-or-assist
 Code for AIES'24 paper "Automate or Assist? The Role of Computational Models in Identifying Gendered Discourse in US Capital Trial Transcripts" 

## Environment
```
conda create -n {env_name} python=3.11
conda activate {env_name}
pip install -r requirements.txt
```

## Steps
1. Finetune legalbert and make predictions
```sh
python3 finetune_legalbert_with_sliding_window.py ${theme_name} ${test_defendant}
```

`theme_name`: one of `Emotions`, `Parent`, `Manipulative`, `Cheating`
`test_defendant`: one value in the `defendant` column in `./data/annotated_transcript.csv`. 

2. Calculate precision, recall, and f1-score for every theme and every defendant (trial). 
```sh
python3 evaluate_predictions.py
```

