"""Feature Wise Transformations Applications — Point d'entrée de l'application Streamlit."""

import streamlit as st

st.set_page_config(page_title="Feature Wise Transformations Applications", page_icon="🧠", layout="wide")

pages = st.navigation([
    st.Page("pages/0_Présentation.py", title="Présentation", default=True),
    st.Page("pages/1_Sort_of_CLEVR.py", title="Sort of CLEVR"),
    st.Page("pages/2_CLEVR_VQA.py", title="CLEVR VQA"),
    st.Page("pages/3_Style_Transfer.py", title="Style Transfer"),
])

pages.run()
