## TODO: rename file
import sys
from pathlib import Path
import json
import pandas as pd

theme = sys.argv[1]
model = "legal-bert-base-uncased"

full = pd.read_csv("./data/maxqda_full_transcript.csv")
coref = pd.read_csv("./data/coref.csv")

pred_true = dict()
precision = list()
for f in Path("./predictions/").glob(f"{model}_sliding_{theme}_*.json"):
    defendant = str(f).split(theme)[-1].replace("_", " ").split(".")[0].strip()
    result = json.load(open(f, 'r'))
    result = pd.DataFrame(result)
    result["sentence_id"] = result["paragraph_idx"].apply(lambda p: p.split(","))
    x = result[["sentence_id", "score"]].explode("sentence_id").groupby(["sentence_id"])["score"].mean().sort_values(ascending=False).reset_index(name='mean_score')
    x['paragraph_id'] = defendant.replace(' ', '_')+"-" + x['sentence_id']
    d_full = full[full["defendant"] == defendant]
    d_coref = coref[coref['defendant'] == defendant]

    out = d_full[['paragraph_id', theme]].merge(x)
    out = d_coref[['coref_in_target', 'target_mentions_defendant', 'paragraph_id', 'paragraph', 'target_sentence']].merge(out)
    out['pred'] = out['mean_score'].apply(lambda t: int(t>=0.5))

    # find consecutive sentences that has positive predictions ("pred" == 1)
    out['value_grp'] = (out['pred'].diff(1) != 0).astype('int').cumsum()
    out['target_mentions_defendant'] = out['target_mentions_defendant'].astype(int)
    out['highlight_sentence'] = out.apply(lambda row: f"<p score={row['mean_score']:.3f}> {row['target_sentence']} </p>" if row['mean_score']>0.9 else row['target_sentence'], axis=1)
    out['highlight_coref'] = out.apply(lambda row: row['target_mentions_defendant'] if row['mean_score'] > 0.9 else None, axis=1) # only document sentence that score > 0.9

    a = out[out['pred'] == 1].groupby(['value_grp'])['highlight_sentence'].apply(lambda t: "\n".join(t))
    b = out[out['pred'] == 1].groupby(['value_grp'])['highlight_coref'].agg("sum")
    c = out[out['pred'] == 1].groupby(['value_grp'])['mean_score'].agg("max")
    d = (out[out['pred'] == 1].groupby(['value_grp'])[theme].agg("sum")>1).astype(int)
    e = out[out['pred'] == 1].groupby(['value_grp'])['sentence_id'].apply(lambda t: ", ".join(t))

    final = a.to_frame().join(b).join(c).join(d).join(e).reset_index()
    final = final[final['highlight_coref']>=1].sort_values("mean_score",ascending=False)
    final['n_sentences'] = final['sentence_id'].apply(lambda s: len(s.split(", ")))
    precision.append({"defendant": defendant, "precision": round(final[final['mean_score']>0.9][theme].sum()/len(final[final['mean_score']>0.9]), 3)})
    # print(f"\tn_paragraphs > 0.9: {len(final[final['mean_score']>0.9])}, n_sentences: {final[final['mean_score']>0.9]['n_sentences'].sum()}, precision: {final[final['mean_score']>0.9][theme].sum()/len(final[final['mean_score']>0.9]):.3f}")
    # print(f"\tprecision @ 3: {final[:3][theme].sum() / 3:.3f}")
    final['pred'] = 1
    final = final.rename({"highlight_sentence": "paragraph", "highlight_coref": "mentions_defendant"}, axis=1)

    pred_true[defendant] = final
print(precision)

# with pd.ExcelWriter(f'./data/{theme}_pred_true_paragraph.xlsx') as writer:  
#     for key, df in pred_true.items():
#         df.to_excel(writer, sheet_name=key)
