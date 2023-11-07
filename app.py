import os
from apikey import apikey
import streamlit as st
from langchain.llms import HuggingFaceHub

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

# Set the OpenAI API key
os.environ['HUGGINGFACEHUB_API_TOKEN'] = apikey

# App framework
st.title('🎮📷🎥 Youtube GPT Creator')
prompt = st.text_input('Plug in your prompt here')

#prompt template
title_template = PromptTemplate(
    input_variables= ['topic'],
    template='write me a youtube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables= ['title','wikipedia_research'],
    template='write me a youtube video script based on the title: {title} while leveraging this wikipedia research:{wikipedia_research}'
)
repo_id = "google/flan-t5-xxl" 

#memory
title_memory = ConversationBufferMemory(input_key='topic',memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title',memory_key='chat_history')


llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.2, "max_length": 256}
)
title_chain=LLMChain(llm=llm,prompt=title_template,verbose=True,output_key= 'title',memory=title_memory)
script_chain=LLMChain(llm=llm,prompt=script_template,verbose=True,output_key = 'script',memory=script_memory)

wiki = WikipediaAPIWrapper()
# Check if the prompt is provided
if prompt:
    
    title= title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script= script_chain.run(title=title,wikipedia_research=wiki_research)
    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)
    
    with st.expander('Message History'):
        st.info(script_memory.buffer)
    
    with st.expander('Wikipedia research History'):
        st.info(wiki_research)