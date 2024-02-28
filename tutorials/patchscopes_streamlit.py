from __future__ import annotations

import streamlit as st

st.title('Patchscopes Streamlit Tutorial')

st.header('Overview')
st.write("""
Patchscopes implements 5 tasks:
1. Decoding next-token predictions
2. Attribute extraction
3. Entity resolution
4. Cross-model patching
5. Multi-hop reasoning

In this tutorial, we will walk through each of these in turn, for a set of mechanistic interpretability lenses.
We will then present our Patchscopes implementation of each lens, and demonstrate how to use it.
Let's get started!
"""
)