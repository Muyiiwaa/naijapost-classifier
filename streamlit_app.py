import streamlit as st
from utils import predict
from config import Settings
import pandas as pd


settings = Settings()


def create_dataframe() -> pd.DataFrame:
    """creates a dataframe of  categories the model was trained on."""
    categories: list = settings.categories
    data: dict = {'categories': categories}
    final_data = pd.DataFrame(data=data)

    return final_data


def get_sidebar():
    st.sidebar.title(body= "Model and App info.")
    st.sidebar.divider()
    if st.sidebar.toggle(label= "All Categories"):
        st.sidebar.dataframe(create_dataframe())
        st.sidebar.divider()


def main():
    st.title(body= "Naija Post Classifier Demo.")
    st.divider()
    get_sidebar()
    text_input = st.text_area(label="Type or Paste text here",
    height = 200)
    if st.button(label="classify text", type="primary"):
        if text_input:
            probability, predicted_class = predict(texts= text_input)
            st.success(f""" This post belongs to the category of {predicted_class},
            with a probability of {probability}""")
        else:
            st.error(f"Text Area cannot be empty! ")

if __name__ == "__main__":
    main()
