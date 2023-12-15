from tiktoken import get_encoding
import tiktoken
from weaviate_interface import WeaviateClient
from prompt_templates import question_answering_prompt_series, question_answering_system
from openai_interface import GPT_Turbo
from app_features import (convert_seconds, generate_prompt_series, search_result,
                          validate_token_threshold, load_content_cache, load_data)
from reranker import ReRanker
from loguru import logger
import streamlit as st
import os

# load environment variables
from dotenv import load_dotenv
load_dotenv('.env', override=True)

## PAGE CONFIGURATION
st.set_page_config(page_title="Impact Theory - Sergio",
                   page_icon=None,
                   layout="wide",
                   initial_sidebar_state="auto",
                   menu_items=None)
##############
# START CODE #
##############
data_path = './data/impact_theory_data.json'

## RETRIEVER --> weaviate_interface.py
api_key = os.environ['WEAVIATE_API_KEY']
url = os.environ['WEAVIATE_ENDPOINT']
client = WeaviateClient(api_key, url)
client.display_properties.append('summary')

## RERANKER --> reranker.py
reranker = ReRanker(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')

## LLM  --> openai_interface.py
model_name = 'gpt-3.5-turbo-0613'
llm = GPT_Turbo(model_name)

## ENCODING  --> tiktoken library
encoding = tiktoken.encoding_for_model(model_name)

## INDEX NAME  --> name of your class on Weaviate cluster
index_name = 'Impact_theory_minilm_256'

##############
#  END CODE  #
##############
data = load_data(data_path)
#creates list of guests for sidebar
guest_list = sorted(list(set([d['guest'] for d in data])))

def main():

    with st.sidebar:
        guest = st.selectbox('Select Guest', options=guest_list, index=None, placeholder='Select Guest')
        show_results = st.toggle('Show search results', True)
        alpha = st.slider('Alpha for hybrid search', 0.0, 1.0, 0.4, 0.1)
        limit = st.slider('Hybrid search retrieval results', 0, 10, 5, 1)
        use_reranker = st.toggle('Use reranker', True)
        if use_reranker:
            top_k = st.slider('Reranker top k', 0, 10, 5, 1)
        do_rag = st.toggle('RAG', False)
        if do_rag:
            temperature = st.slider('Temperature of LLM', 0.0, 1.0, 0.0, 0.1)

    st.image('./assets/impact-theory-logo.png', width=400)
    st.subheader(f"Chat with the Impact Theory podcast: ")
    st.write('\n')
    col1, _ = st.columns([7,3])
    # with col1:
    query = st.chat_input('Enter your question: ') if do_rag \
        else st.text_input('Enter your question: ')

    # with st.chat_message('assistant'):

    if query:
        # make hybrid call to weaviate
        where_filter = {
            "path": ["guest"],
            "operator": "Equal",
            "valueText": guest
        } if guest else None

        hybrid_response = client.hybrid_search(query, index_name, alpha=alpha, limit=limit, where_filter=where_filter)

        # rerank results
        ranked_response = reranker.rerank(hybrid_response, query, apply_sigmoid=True, top_k=top_k) if use_reranker else hybrid_response

        # validate token count is below threshold
        valid_response = validate_token_threshold(ranked_response,
                                                    question_answering_prompt_series,
                                                    query=query,
                                                    tokenizer=encoding,
                                                    token_threshold=3500,
                                                    verbose=True)

        if show_results:
            st.subheader("Search Results")
            for i, hit in enumerate(valid_response):
                col1, col2 = st.columns([7, 3], gap='large')
                image = hit['thumbnail_url'] # get thumbnail_url
                episode_url = hit['episode_url'] # get episode_url
                title = hit['title'] # get title
                show_length = hit['length'] # get length
                time_string = convert_seconds(show_length) # convert show_length to readable time string

                with col1:
                    st.write( search_result(  i=i,
                                                url=episode_url,
                                                guest=hit['guest'],
                                                title=title,
                                                content=hit['content'],
                                                length=time_string),
                            unsafe_allow_html=True)
                    st.write('\n\n')
                with col2:
                    st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)

                st.markdown("----")

        if do_rag:
            st.chat_message('user').markdown(query)

            with st.chat_message('assistant'):
                # prep for streaming response
                # st.subheader("Response from Impact Theory")
                with st.spinner('Generating Response...'):

                    # creates container for LLM response
                    chat_container, response_box = [], st.empty()

                    # generate LLM prompt
                    prompt = generate_prompt_series(query=query, results=valid_response)

                    # execute chat call to LLM
                    response = llm.get_chat_completion(prompt,
                                                        temperature=temperature,
                                                        show_response=True,
                                                        max_tokens=500,
                                                        stream=True)

                    # iterate through the stream of events
                    for chunk in response:
                        try:
                            # inserts chat stream from LLM
                            with response_box:
                                content = chunk.choices[0].delta.content
                                if content:
                                    chat_container.append(content)
                                    result = "".join(chat_container).strip()
                                    st.markdown(f'{result}'+ "▌")
                        except Exception as e:
                            print(e)
                            continue
                    response_box.markdown(result)

if __name__ == '__main__':
    main()