import streamlit as st
from gramformer import Gramformer
import torch
st.set_page_config(layout="wide")


def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(1212)

st.title("Correct your text here:")

form = st.form(key='Grammar Correction')
fi_incorrect = form.text_input(label="**:blue[Write your text here:]**", value="This is a test text.")
btn = form.form_submit_button(label='Submit')

gf = Gramformer(models=1, use_gpu=False) # 1=corrector, 2=detector


influent_sentences = [
    fi_incorrect
]   

for influent_sentence in influent_sentences:
    corrected_sentences = gf.correct(influent_sentence, max_candidates=1)
    for corrected_sentence in corrected_sentences:

        col1, col2 = st.columns(2)

        with col1:
            st.header("Your Input:")
            st.write(fi_incorrect)

        with col2:
            st.header("Corrected Text:")
            st.write(corrected_sentence)

    print("-" *100)

