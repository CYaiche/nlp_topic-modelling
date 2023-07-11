import streamlit as st
import pandas as pd
from predict import *

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
            use_topic_model = TopicModellingModel()
            concatenation = user_title + user_body
            tags = use_topic_model.run_inference(concatenation)
            st.markdown("""
                <style>
                .big-font {
                    font-size:50px !important;
                    color: blue;
                }
                </style>
                """, unsafe_allow_html=True)
            st.markdown('# Tags : ', unsafe_allow_html=True)
            
            for tag in tags : 

                st.markdown(f'<p class="big-font">{tag}</p>', unsafe_allow_html=True)
if __name__ == "__main__" : 
    main()