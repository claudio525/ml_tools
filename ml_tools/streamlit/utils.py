import streamlit as st


def update_st_width(
    max_width: int,
    padding_top: int,
    padding_bottom: int,
    padding_right: int,
    padding_left: int,
):
    """
    Updates the with and padding of the streamlit app

    Parameters
    ----------
    max_width: int
        Maximum width of the app
    padding_top: int
        Padding at the top
    padding_bottom: int
        Padding at the bottom
    padding_right: int
        Padding at the right
    padding_left: int
        Padding at the left
    """
    st.markdown(
        f"""
    <style>
        .appview-container .main .block-container{{
            max-width: {max_width}px;
            padding-top: {padding_top}rem;
            padding-right: {padding_right}rem;
            padding-left: {padding_left}rem;
            padding-bottom: {padding_bottom}rem;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )
