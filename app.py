import streamlit as st
from pinecone import ServerlessSpec
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
from datasets import load_dataset
from io import BytesIO
from base64 import b64encode
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# Load Pinecone configuration
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_CLOUD = os.getenv('PINECONE_CLOUD')
PINECONE_REGION = os.getenv('PINECONE_REGION')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')

# Initialize Pinecone client and index
@st.cache_resource
def initialize_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    return index

index = initialize_pinecone()

# Load dataset
@st.cache_resource
def load_fashion_dataset():
    fashion = load_dataset("ashraq/fashion-product-images-small", split="train")
    images = fashion["image"]
    metadata = fashion.remove_columns("image")
    return images, metadata

images, metadata = load_fashion_dataset()

product_df = metadata.to_pandas()

# Load models
@st.cache_resource
def load_models():
    doc_model_id = "naver/efficient-splade-VI-BT-large-doc"
    doc_tokenizer = AutoTokenizer.from_pretrained(doc_model_id)
    doc_model = AutoModelForMaskedLM.from_pretrained(doc_model_id)

    query_model_id = "naver/efficient-splade-VI-BT-large-query"
    query_tokenizer = AutoTokenizer.from_pretrained(query_model_id)
    query_model = AutoModelForMaskedLM.from_pretrained(query_model_id)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dense_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device=device)

    return doc_tokenizer, doc_model, query_tokenizer, query_model, dense_model

doc_tokenizer, doc_model, query_tokenizer, query_model, dense_model = load_models()

# Define utility functions
def compute_vector(text, tokenizer, sparse_model):
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    output = sparse_model(**tokens)
    logits, attention_mask = output.logits, tokens.attention_mask
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    max_val, _ = torch.max(weighted_log, dim=1)
    vec = max_val.squeeze()
    return vec, tokens

def hybrid_scale(dense, query_indices, query_values, alpha: float):
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    query_indices_list = query_indices.tolist()
    query_values_list = query_values.tolist()
    dense_list = dense.tolist()
    hsparse = {
        'indices': query_indices_list,
        'values': [v * (1 - alpha) for v in query_values_list]
    }
    hdense = [v * alpha for v in dense_list]
    return hdense, hsparse


def create_docs(results):
  docs = []
  for i,result in enumerate(results['matches']):
    #print(result)
    pid = int(result['metadata']['id'])
    #print(pid)
    score = result['score']
    #print(score)
    result_string = ""
    #
    product_name = product_df[product_df['id'] == pid][['productDisplayName']].values[0].tolist()[0]
    category = product_df[product_df['id'] == pid][["masterCategory"]].values[0].tolist()[0]
    article = product_df[product_df['id'] == pid][["articleType"]].values[0].tolist()[0]
    usage = product_df[product_df['id'] == pid][["usage"]].values[0].tolist()[0]
    season = product_df[product_df['id'] == pid][["season"]].values[0].tolist()[0]
    gender = product_df[product_df['id'] == pid][["gender"]].values[0].tolist()[0]
    #
    result_string += "Product Name:" +product_name+";" + "Category:" +category+";" + "Article Type:"+article+";"\
    "Usage:" + usage + ";" + "Season:" + season+ ";"+ "Gender:" + gender
    #
    doc = Document(page_content = result_string)
    doc.metadata['pid'] = str(pid)
    doc.metadata['score'] = score
    docs.append(doc)
  return docs

# Streamlit UI
st.title("E-commerce Search - Hybrid")

searchquery = st.text_input("Search for the product")
alpha_input = st.text_input("Enter between 0 and 1 where 0 == sparse only and 1 == dense only")
search_bt = st.button("Search")

if search_bt:
    print(searchquery)
    print(alpha_input)
    question=searchquery
    dense_vec = dense_model.encode(searchquery)
    query_vec, query_tokens = compute_vector(searchquery, query_tokenizer, query_model)
    query_indices = query_vec.nonzero().numpy().flatten()
    query_values = query_vec.detach().numpy()[query_indices]
    hdense, hsparse = hybrid_scale(dense_vec, query_indices, query_values, alpha=float(alpha_input))

    # Search the index
    result = index.query(
        top_k=14,
        vector=hdense,
        sparse_vector=hsparse,
        include_metadata=True
    )

    # Use returned product IDs to get images
    imgs = [images[int(r["id"])] for r in result["matches"]]

    #print(result['matches'][0])

    docs = create_docs(result)
    #print(docs)

    #
    template = """
    You are a fashion shopping assistant aiming to persuade customers using the provided information. Describe the season and usage mentioned in the context in your interaction with the customer.
    Utilize a bullet list to detail each product for the customer.

    Context : {context}
    User question :{question}
    Your Response:"""
    #
    prompt = PromptTemplate.from_template(template)
    print(prompt)
    #
    chain = load_qa_chain(llm=ChatOpenAI(model_name="gpt-4",
                                        temperature=0.5),
                        chain_type='stuff',
                        prompt=prompt)
    #
    response = chain({"input_documents":docs,
                    "question":question},
                    return_only_outputs=True)
    #
    llm_output = response['output_text']
    print(llm_output)

    # Display the result text
    st.write("## Recommended Products")
    st.write(llm_output)

    # Display images in rows of 3
    images_per_row = 3
    for i in range(0, len(result["matches"]), images_per_row):
        cols = st.columns(images_per_row)
        for j, col in enumerate(cols):
            image_index = i + j
            if image_index < len(result["matches"]):
                match = result["matches"][image_index]
                img = imgs[image_index]
                product_name = match["metadata"]['productDisplayName']
                print(product_name)
                with col:
                    st.image(img, caption=product_name)
