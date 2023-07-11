import streamlit as st
import pandas as pd


def main() : 
    st.markdown(""" #  ML deployment  """)
    st.header("""Clara Yaiche - OpenClassrooms NLP project""")
    st.write('The app tags stackoverflow posts. The classification can output 0 to 30 different tags.\
    You can try to write a new posts or search online and try existing ones. ')
    user_title = st.text_input("Enter the title of your post")
    
    user_body = st.text_area("Enter the message of your post (minimum 100 characters)", height=200)
    

        
    if st.button('Tag the post'):
        if  len(user_body) < 100:
            st.warning("Please enter at least 100 characters.")
        else : 
            st.write("OK")

if __name__ == "__main__" : 
    main()