import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
HF_TOKEN=os.getenv("HF_TOKEN")
st.write(f"API Key Loaded: {groq_api_key is not None}")
# Set up the embeddings
try:
    # If you have a specific Hugging Face model, you can use it like this
   
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # Add API key if needed
except Exception as e:
    st.error(f"Failed to load embeddings: {e}")
    embeddings = None

data = pd.read_csv("Dataset.csv")

# Convert to a format compatible with LangChain
documents = data.to_dict(orient="records")

# Create the vector store
vector_store = FAISS.from_texts(
    [doc["Description"] for doc in documents],
    embeddings,
    metadatas=[{"Course Name": doc["Course Name"], "Video Link": doc["Resource Link"], "Price": doc["Price"]} for doc in documents]
)

# Set up the retriever
retriever = vector_store.as_retriever()


# Set up the LLM (Groq model)
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=os.getenv("GROQ_API_KEY"))

# # Define the prompt template
# message = """
# You are the owner of a company selling online courses. The user will give information about the course. You have to give the description of the course, course name,resource link and the price . 
# Answer this question using the provided context only.

# {question}

# Context:
# {context}
# """

# prompt = ChatPromptTemplate.from_messages([("human", message)])

# # Define the RAG chain
# rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

# # Streamlit UI
# st.title('Build Fast With AI 🚀')

# st.write("### Enter Course Description")
# course_description = st.text_input("Describe the course you are looking for:")

# if st.button("Find Course"):
#     if course_description:
#         response = rag_chain.invoke(course_description)
#         st.write("### Recommended Course Information")
#         st.write(response.content)
#         #st.write(f"**Course Description**: {response.content['Description']}")
#         #st.write(f"**Video Link**: [Click here]({response.content['Resource Link']})")
#     else:
#         st.write("Please enter a course description.")
from langchain.chains import RetrievalQA
search_prompt_template = """
You are a search assistant with access to a course database.
Given a user query.

First Search the user query in the name of the course if the course name is found similar to user query than return information for tha course.
else  use the database to find the **best match** course.
Return only the following information in a structured format:

- **Course Name**: {Course Name}
- **Course Description**: {Description}
- **Course Resource Link**: {Video Link}

Only provide one best match based on the user query.

Query: {query}
"""

# Create a RetrievalQA Chain (LLM guides retrieval)
retrieval_chain = RetrievalQA.from_chain_type(
    retriever=retriever,
    llm=llm,
    chain_type="stuff",
    return_source_documents=True#metadata
)

# Streamlit UI
st.title('Smart Course Finder with AI 🚀')
course_query = st.text_input("Describe the course you are looking for:")

if st.button("🔍 Search for Course"):
    if course_query:
        with st.spinner("Searching for the best match..."):
            result = retrieval_chain.invoke(course_query)
            
            if result and "source_documents" in result and result["source_documents"]:
               
                best_match = result["source_documents"][0].metadata
                course_name = best_match.get("Course Name", "N/A")
                video_link = best_match.get("Video Link", "N/A")
                price = best_match.get("Price", "N/A")
            
                st.write(f"**Course Name**: {course_name}")
                st.write(f"**Description**: {result["source_documents"][0].page_content}")
                st.write(f"**Resources**: [Click here]({video_link})")
                st.write(f"**Price**: {price}")
            else:
                st.write("No matching course found.")
    else:
        st.write("Please enter a query.")
