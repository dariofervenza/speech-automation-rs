# Speech automation Rust

Load audio wav, resample it and extract speech using Nvidia canary 1b v2


## Future goals

What I want to build is an app that:

- Continously listens to my microphone

- Extract many audio temporal windows

- Performs speech recognition

- Searches for specific commands that I will code

- Executes the command when a match is found

- Registers in MongoDb each interaction and the parsed speech

- Containerized application, easy to deply


## Models

- [Canary v2 ONNX](https://huggingface.co/istupakov/canary-1b-v2-onnx)

## TODOS:

- Do not hardcode --> Move everything to config/specs files

- Add tests

- Expand with more models?


# Why I am creating this

I wanted to practice Rust but my profile is focused on Python.

I could start a pytorch project or anything python-based.

However, using rust I'm going to achieve many goals:

- Handle problems in the hard way + learn more Rust

- Learn about model preprocessing tehcniques + SOTA models (eg. audio preprocessing with fbanks)

- Maintain contact with python projects due to the rust complexity and the necessity to reverse-engineer model architectures

- Practice MongoDB, as it will my main database
