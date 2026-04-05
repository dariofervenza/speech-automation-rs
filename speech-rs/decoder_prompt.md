- Load the model with python.
pip install -U nemo_toolkit['asr']
from nemo.collections.asr.models import ASRModel
asr_ast_model = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")

- Inspect which model are we using, which tokenizer, which promptformatter

print(asr_ast_model.__class__)

print(asr_ast_model.tokenizer)
print(asr_ast_model.tokenizer.tokenizer)

- Get the tokenizer map

print(list(asr_ast_model.tokenizer.tokenizer.get_vocab().items())[:1000])
print(asr_ast_model.tokenizer.tokenizer.get_vocab())

with open("tokenizer_export.json", "w", encoding="utf-8") as f:
    json.dump(asr_ast_model.tokenizer.tokenizer.get_vocab(), f)

- Which prompt are using?

Check nemo/collecitons/asr/models/aed_multitask_models.py -> EndDecMultiTaskModel
Check nemo/collections/common/prompts/formatter
Check nemo/collecitons/common/prompts/canary2.py

print(asr_ast_model.prompt)

pf = asr_ast_model.prompt
turns = [
    {
        "role": "user",
        "slots": {
            "source_lang": "en",
            "target_lang": "en",
            "decodercontext": "Please transcribe this audio file in english",
            "emotion": "<|emo:undefined|>",
            "pnc": "yes",
            "itn": "no",
            "timestamp": "no",
            "diarize": "no",
        },
    },
    {
        "role": "user_partial,
        "slots": {
            "decodercontext": "Please go ahead with the transcription
        }
    }
]
encoded = pf.encode_dialog(turns)
print(encoded)
ids = encoded["input_ids"].tolist()
text = asr_ast_model.tokenizer.ids_to_text(ids)
print(text)
tokens = asr_ast_model.tokenizer.ids_to_tokens(ids)
print(tokens)


- Result:

{'input_ids': tensor([16053,     7,     4,    16,    64,    64,     5,     9,    11,    13]), 'context_ids': tensor([16053,     7,     4,    16,    64,    64,     5,     9,    11,    13])}


['▁', '<|startofcontext|>', '<|startoftranscript|>', '<|emo:undefined|>', '<|en|>', '<|en|>', '<|pnc|>', '<|noitn|>', '<|notimestamp|>', '<|nodiarize|>']
