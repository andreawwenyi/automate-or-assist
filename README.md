# automate-or-assist
 Code for AIES'24 paper "Automate or Assist? The Role of Computational Models in Identifying Gendered Discourse in US Capital Trial Transcripts" 

## Environment
```
conda create -n {env_name} python=3.11
conda activate {env_name}
pip install -r requirements.txt
```
## Data
To run the scripts, please provide the following 2 data files. 

1. `./data/annotated_transcripts.csv`

This file contains all the sentences from all the trial transcripts. Each row is a sentence from a trial transcript. 

Schema:
| Column Name | Type | Format | Explanation |
| ------------- | ------------- | ------------- |------------- |
| paragraph_id  | str | `{defendant name}-{sentence index in the trial}` | This is the primary key of the csv. It serves as a unique identifier of a sentence. |
| defendant  | str | `{defendant name}` | This column serves as a trial identifier.|
| text  | str  | `{sentence}` | The sentence. Lower-cased.|
| {theme_name_1}  | int |`1` or `0`| `1` if the sentence is annotated as belongs to {theme_name_1} else 0.|
| {theme_name_2}  | int |`1` or `0`| `1` if the sentence is annotated as belongs to {theme_name_2} else 0.|
| {theme_name_n}  | int |`1` or `0`| `1` if the sentence is annotated as belongs to {theme_name_n} else 0.|

2. `./data/nicknames.json`

This file is to document ways a specific defendant could be mentioned with other nicknames. This is used in `run_coreference_resolution.py` to identify whether a name refers to the defendant. If the defendant doesn't have nicknames, there is no need to add the defendant in this json file. 

The keys of the json have to match the values of `defendant` column in `./data/annotated_transcripts.csv`.
```
{
    {defendant_A name}: [
        {defendant_A's nickname 1}, {defendant_A's nickname 2}, ...
    ],
    {defendant_B name}: [
        {defendant_B's nickname 1}, {defendant_B's nickname 2}, ...
    ],
    ...
}
```

## Steps
1. Finetune LEGAL-BERT and make predictions:
```sh
python3 finetune_legalbert.py ${theme_name} ${test_defendant}
```

* `theme_name`: one of `{theme_name_1}`, `{theme_name_2}`, ..., `{theme_name_n}` in `./data/annotated_transcripts.csv`.
* `test_defendant`: one value in the `defendant` column in `./data/annotated_transcripts.csv`. 

2. Run coreference resolution:
```sh
python3 run_coreference_resolution.py
```

3. Combine theme prediction and coreference resolution results:
```sh
python3 combine_pred_and_coref.py ${theme_name}
```

* `theme_name`: one of `{theme_name_1}`, `{theme_name_2}`, ..., `{theme_name_n}` in `./data/annotated_transcripts.csv`.