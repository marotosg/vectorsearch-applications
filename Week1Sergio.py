#standard libraries
import json
import os
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Union, Callable
from math import ceil
import tiktoken # bad ass tokenizer library for use with OpenAI LLMs 

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

#external libraries
import pandas as pd
import numpy as np
#from rich import print
#from torch import cuda
from tqdm.notebook import tqdm

#external files
from preprocessing import FileIO

#root folder on Google Colab is: /content/
root_folder = './data/'
data_file = 'impact_theory_data.json'
data_path = os.path.join(root_folder, data_file)

with open(data_path) as f:
    data =  json.load(f)
print(f'Total # of episodes: {len(data)}')

contents = [d['content'] for d in data]
content_lengths = [len(content.split()) for content in contents]
df = pd.DataFrame(content_lengths, columns=['# Words'])
print(df.describe())

mean_word_count = ceil(np.mean(content_lengths))
token_to_word_ratio = 1.3
approx_token_count = ceil(mean_word_count * token_to_word_ratio)
print(f'The mean word count for each episode is about {mean_word_count} words, which corresponds to a rough token count of {approx_token_count} tokens.')

#instantiate tokenizer for use with ChatGPT-3.5-Turbo
encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-0613')

tokens = encoding.encode_batch(contents)
token_counts = list(map(len, tokens))
token_df = pd.DataFrame(token_counts, columns=['# Tokens'])
print(token_df.describe())

true_ratio = round(np.mean(token_counts)/mean_word_count, 2)
#pretty close...
print(true_ratio)
total_tokens = sum(token_counts)
print(f'Total Tokens in Corpus: {total_tokens}')

from llama_index.text_splitter import SentenceSplitter #one of the best on the market

#set chunk size and instantiate your SentenceSplitter
chunk_size = 256
gpt35_txt_splitter = SentenceSplitter(chunk_size=chunk_size, tokenizer=encoding.encode, chunk_overlap=0)

def split_contents(corpus: List[dict],
                   text_splitter: SentenceSplitter,
                   content_field: str='content'
                   ) -> List[List[str]]:
    '''
    Given a corpus of "documents" with text content, this function splits the
    content field into chunks sizes as specified by the text_splitter.

    Example
    -------
    corpus = [
            {'title': 'This is a cool show', 'content': 'There is so much good content on this show. \
              This would normally be a really long block of content. ... But for this example it will not be.'}, 
            {'title': 'Another Great Show', 'content': 'The content here is really good as well.  If you are \
              reading this you have too much time on your hands. ... More content, blah, blah.'}
           ]
           
    output = split_contents(data, text_splitter, content_field="content")
    
    output >>> [['There is so much good content on this show.', 'This would normally be a really long block of content.', \
                 'But for this example it will not be'], 
                ['The content here is really good as well.', 'If you are reading this you have too much time on your hands.', \
                 'More content, blah, blah.']
                ]
    '''
    corpus_size = len(data)
    list_sentences = []
    for i in tqdm(range(int(corpus_size)), desc="Progress"):
        sentence = corpus[i]
        sentence_split = text_splitter.split_text(sentence)
        list_sentences.append(sentence_split)
    return list_sentences
    ########################
    # START YOUR CODE HERE #
    ########################


content_splits = split_contents(contents, gpt35_txt_splitter)

def get_split_lengths(splits: List[List[str]], column_name: str='Split Lengths') -> pd.DataFrame:
    '''
    Given a list of text splits, returns the length of each split
    in a pandas DataFrame.
    '''
    lengths = list(map(len, splits))
    return pd.DataFrame(lengths, columns=[column_name])


column_name = 'Split Lengths'
# replace None with the output from the split_contents function
df = get_split_lengths(content_splits, column_name=column_name)

# reverse the order of the episode # to correctly show left to right chronological order
df.index = sorted(list(df.index), reverse=True)
# create plot
ax = df.iloc[::-1].plot.bar(y='Split Lengths', xlabel='Episode #', ylabel='Approx. Length in Tokens', title='Episode Length (in tokens) over Time', figsize=(20,8))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

#define the model you want to use
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

#create sample sentences
text1 = "I ran down the road"
text2 = "I ran down the street"
text3 = "I ran down the lane"
text4 = "I ran down the avenue"
text5 = "I ran over to the house"

road_sentences = [text1, text2, text3, text4, text5]
road_vectors = model.encode(road_sentences)
road_vectors, road_vectors.shape

# we need to reencode our texts here with pytorch tensors to work with the semantic_search function
tensors = model.encode(road_sentences, convert_to_tensor=True)

# we'll set our query to the first sentence
query = model.encode(text1, convert_to_tensor=True)

#compare our query with all of the other sentences, including itself (expect to see a cossim value of 1)
results = semantic_search(query_embeddings=query, corpus_embeddings=tensors)[0]
for result in results:
    print(f'Score: {round(result["score"],3)} - {road_sentences[result["corpus_id"]]}')

    #define some semantically similar sentences 
passages = ['Tom Bilyeu is the host of Impact Theory and has helped millions achieve their dreams',
            'Tom Bilyeu founded Quest Nutrition in 2010',
            'Tom Bilyeu, not known outside of those who listen to the Impact THeory',
            'Tom Bilyeu claims to have deadlifted 335 pounds on his show',
            'Tomcats are not neutered cats']

#we'll define a query that should be able to be answered by the passages
query = "Who is Tom Bilyeu"

tom_tensors = model.encode(passages, convert_to_tensor=True)
tom_query = model.encode(query, convert_to_tensor=True)
tom_results = semantic_search(query_embeddings=tom_query, corpus_embeddings=tom_tensors)
for result in tom_results[0]:
    print(f'Score: {round(result["score"],3)} - {passages[result["corpus_id"]]}')

def encode_content_splits(content_splits: List[List[str]],
                          model: SentenceTransformer,
                          device: str='cuda:0'
                          ) -> List[List[Tuple[str, np.array]]]:
    ########################
    # START YOUR CODE HERE #
    ########################
    content_size = len(content_splits)
    list_vectors = []
    for i in tqdm(range(int(content_size)), desc="Progress embedding"):
            list_embedding = []
            episode_content = content_splits[i]
            for part_episode  in episode_content:
                 episode_vector = model.encode(part_episode, convert_to_tensor=True)
                 tuple_part_episode = tuple([part_episode,episode_vector])
                 list_embedding.append(tuple_part_episode)
            list_vectors.append(list_embedding)
    return list_vectors 

text_vector_tuples = encode_content_splits(content_splits, model)  

def join_metadata(corpus: List[dict], 
                  text_vector_list: List[List[Tuple[str, np.array]]],
                  content_field: str='content',
                  embedding_field: str='content_embedding'
                 ) -> List[dict]:
    '''
    Combine episode metadata from original corpus with text/vectors tuples.
    Creates a new dictionary for each text/vector combination.
    '''
    ########################
    # START YOUR CODE HERE #
    ########################
    corpus_size = len(corpus)
    joined_documents = []
    for i in tqdm(range(int(corpus_size)), desc="Creating corpus"):
        episode = corpus[i]
        inner_vector_list = text_vector_list[i]
        ran = range(len(inner_vector_list))
        for j in ran:
            tuple_part_episode = inner_vector_list[j]
            part_text = tuple_part_episode[0]
            part_vector = tuple_part_episode[1]
            vector_array = list(part_vector.flatten())
            doc_id = episode.get("video_id")+ "_" + str(j)
            document = dict(doc_id=doc_id,content=part_text,content_embedding=vector_array)
            joined_documents.append(document)
   
    return joined_documents

docs = join_metadata(data, text_vector_tuples)

io = FileIO()

#Define your output path
outpath = "./impact-theory-minilmL6-256.parquet"

#save to disk
io.save_as_parquet(file_path=outpath, data=docs ,overwrite=True)