import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time
import random
import altair as alt


st.markdown("__*CÓDIGO 1:*__")
st.text('This will appear first')
# Appends some text to the app.

my_slot1 = st.empty()
# Appends an empty slot to the app. We'll use this later.

my_slot2 = st.empty()
# Appends another empty slot.

st.text('This will appear last')
# Appends some more text to the app.

my_slot1.text('This will appear second')
# Replaces the first empty slot with a text string.

my_slot2.line_chart(np.random.randn(20, 2))
# Replaces the second empty slot with a chart.

st.markdown("------")
st.markdown("__*CÓDIGO 2:*__")

chart_data = pd.DataFrame(
   np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
   columns=['lat', 'lon'])

st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=37.76,
        longitude=-122.4,
        zoom=11,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
           'HexagonLayer',
           data=chart_data,
           get_position='[lon, lat]',
           radius=200,
           elevation_scale=4,
           elevation_range=[0, 1000],
           pickable=True,
           extruded=True,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=chart_data,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=200,
        ),
    ],
))

st.markdown("------")
st.markdown("__*CÓDIGO 3:*__")

col1, col2, col3 = st.columns(3)

with col1:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg")

with col2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg")

with col3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg")


st.markdown("------")
st.markdown("__*CÓDIGO 4:*__")

from datetime import datetime

start_time = st.slider(
    "When do you start?",
    value=datetime(2020, 1, 1, 9, 30),
    format="MM/DD/YY - hh:mm")
st.write("Start time:", start_time)

st.markdown("------")
st.markdown("__*CÓDIGO 5:*__")

st.metric(label="Temperature", value="70 °F", delta="1.2 °F")

st.markdown("------")
st.markdown("__*CÓDIGO 6:*__")

df = pd.DataFrame(
    [
       {"command": "st.selectbox", "rating": 4, "is_widget": True},
       {"command": "st.balloons", "rating": 5, "is_widget": False},
       {"command": "st.time_input", "rating": 3, "is_widget": True},
   ]
)
edited_df = st.data_editor(df)

favorite_command = edited_df.loc[edited_df["rating"].idxmax()]["command"]
st.markdown(f"Your favorite command is **{favorite_command}** 🎈")

st.markdown("------")
st.markdown("__*CÓDIGO 7:*__")

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["col1", "col2", "col3"])

st.area_chart(
   chart_data, x="col1", y=["col2", "col3"], color=["#FF0000", "#0000FF"]  # Optional
)

st.markdown("------")
st.markdown("__*CÓDIGO 8:*__")

name = st.text_input('Name')
if not name:
  st.warning('Please input a name.')
  st.stop()
st.success('Thank you for inputting a name.')


st.markdown("------")
st.markdown("__*CÓDIGO 9:*__")
st.markdown("sidebar")

with st.sidebar:
    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        messages.chat_message("user").write(prompt)
        messages.chat_message("assistant").write(f"Echo: {prompt}")


st.markdown("------")
st.markdown("__*CÓDIGO 10:*__")

with st.status("Downloading data..."):
    st.write("Searching for data...")
    time.sleep(2)
    st.write("Found URL.")
    time.sleep(1)
    st.write("Downloading data...")
    time.sleep(1)

st.button('Rerun')

st.markdown("------")
st.markdown("__*CÓDIGO 11:*__")

if st.button('Three cheers'):
    st.toast('Hip!')
    time.sleep(.5)
    st.toast('Hip!')
    time.sleep(.5)
    st.toast('Hooray!', icon='🎉')

st.markdown("------")
st.markdown("__*CÓDIGO 12:*__")

def cook_breakfast():
    msg = st.toast('Gathering ingredients...')
    time.sleep(1)
    msg.toast('Cooking...')
    time.sleep(1)
    msg.toast('Ready!', icon = "🥞")

if st.button('Cook breakfast'):
    cook_breakfast()

st.markdown("------")
st.markdown("__*CÓDIGO 13:*__")

df = pd.DataFrame(np.random.randn(10, 20), columns=("col %d" % i for i in range(20)))

st.dataframe(df.style.highlight_max(axis=0))



st.markdown("------")
st.markdown("__*CÓDIGO 14:*__")

df = pd.DataFrame(
    {
        "name": ["Roadmap", "Extras", "Issues"],
        "url": ["https://roadmap.streamlit.app", "https://extras.streamlit.app", "https://issues.streamlit.app"],
        "stars": [random.randint(0, 1000) for _ in range(3)],
        "views_history": [[random.randint(0, 5000) for _ in range(30)] for _ in range(3)],
    }
)
st.dataframe(
    df,
    column_config={
        "name": "App name",
        "stars": st.column_config.NumberColumn(
            "Github Stars",
            help="Number of stars on GitHub",
            format="%d ⭐",
        ),
        "url": st.column_config.LinkColumn("App URL"),
        "views_history": st.column_config.LineChartColumn(
            "Views (past 30 days)", y_min=0, y_max=5000
        ),
    },
    hide_index=True,
)

st.markdown("------")
st.markdown("__*CÓDIGO 15:*__")

def load_data():
    return pd.DataFrame(
        {
            "first column": [1, 2, 3, 4],
            "second column": [10, 20, 30, 40],
        }
    )

# Boolean to resize the dataframe, stored as a session state variable
st.checkbox("Use container width", value=False, key="use_container_width")

df = load_data()

# Display the dataframe and allow the user to stretch the dataframe
# across the full width of the container, based on the checkbox value
st.dataframe(df, use_container_width=st.session_state.use_container_width)



st.markdown("------")
st.markdown("__*CÓDIGO 16:*__")

with st.chat_message("user"):
    st.write("Hello 👋")
    st.line_chart(np.random.randn(30, 3))

st.markdown("------")
st.markdown("__*CÓDIGO 17:*__")

row1 = st.columns(3)
row2 = st.columns(3)

for col in row1 + row2:
    tile = col.container(height=120)
    tile.title(":balloon:")

st.markdown("------")
st.markdown("__*CÓDIGO 18:*__")

with st.container():
   st.write("This is inside the container")

   # You can call any Streamlit command, including custom components:
   st.bar_chart(np.random.randn(50, 3))

st.write("This is outside the container")

st.markdown("------")
st.markdown("__*CÓDIGO 19:*__")

df = pd.DataFrame(
    np.random.randn(200, 3),
    columns=['a', 'b', 'c'])

c = alt.Chart(df).mark_circle().encode(
    x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])

st.write(c)

st.markdown("------")
st.markdown("__*CÓDIGO 20:*__")

data_df = pd.DataFrame(
    {
        "sales": [200, 550, 1000, 80],
    }
)

st.data_editor(
    data_df,
    column_config={
        "sales": st.column_config.ProgressColumn(
            "Sales volume",
            help="The sales volume in USD",
            format="$%f",
            min_value=0,
            max_value=1000,
        ),
    },
    hide_index=True,
)