#!/usr/bin/env bash

# python scripts/token_identity.py --model gpt2
python scripts/token_identity.py --model mamba
python scripts/token_identity.py --model gptj
python scripts/token_identity.py --model mistral
