import streamlit as st
import pandas as pd
import numpy as np 
import random

data1 = {
    "Col1": np.random.rand(4),
    "Col2": np.random.rand(4),
    "Col3": np.random.rand(4),
    "Col4": np.random.rand(4),
}

df1 = pd.DataFrame(data1)

st.dataframe(df1)

# ----------------------------------------------------------------------------------------------------------

data2 = {
    "Col1": np.random.rand(4),
    "Col2": np.random.rand(4),
    "Col3": np.random.rand(4),
    "Col4": np.random.rand(4),
}

df2 = pd.DataFrame(data2)

# in this case i use style.highlight_max(axis=0) what color in yellow the max cell from each column
# axis = 0 -> is for column
# axis = 1 -> is for row
st.dataframe(df2.style.highlight_max(axis=0))

# ----------------------------------------------------------------------------------------------------------

data3 = {
    "Col1": np.random.rand(4),
    "Col2": np.random.rand(4),
    "Col3": np.random.rand(4),
    "Col4": np.random.rand(4),
}

# here is the dataframe with the content that will be on the table
df3 = pd.DataFrame(
    {
        "name": ["Roadmap", "Extras", "Issues"],
        "url": ["https://roadmap.streamlit.app", "https://extras.streamlit.app", "https://issues.streamlit.app"],
        "stars": [random.randint(0, 1000) for _ in range(3)],
        "views_history": [[random.randint(0, 5000) for _ in range(30)] for _ in range(3)],
    }
   )

st.dataframe(
        df3,
        column_config={
        "name": st.column_config.Column(
            "App name", # the name of the column header
            help="The name of the topics" # on hover will appear that text
        ),
        "stars": st.column_config.NumberColumn(
            "GitHub Stars", # the name of the column header
            help="Number of stars on GitHub", # on hover will appear that text
            format="%d",
        ),
        "url": st.column_config.LinkColumn(
            "App URL", # the name of the column header
            help="The URLs of the topics"), # on hover will appear that text
        "views_history": st.column_config.LineChartColumn(
            "Views (past 30 days)", y_min=0, y_max=5000
        )
    }, hide_index= True)



# ----------------------------------------------------------------------------------------------------------


# use_container_width

# Cache the dataframe so it's only loaded once
@st.cache_data
def load_data():
    return pd.DataFrame(
    {
        "Col1": np.random.rand(4),
        "Col2": np.random.rand(4),
        "Col3": np.random.rand(4),
        "Col4": np.random.rand(4),
    }
)

# Boolean to resize the dataframe, stored as a session state variable
st.checkbox("Use container width", value=False, key="use_container_width")   

df4 = load_data()

st.dataframe(df4, use_container_width=st.session_state.use_container_width)