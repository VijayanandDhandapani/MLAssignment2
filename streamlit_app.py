import streamlit as st

st.title("Welcome to My Streamlit App")
ip = st.text_input("Enter your name:", key="name_input")
st.write(f"Hello, {ip}!")

import time

'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'
