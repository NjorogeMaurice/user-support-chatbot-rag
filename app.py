# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import transformers as tf
# # Initialize the FastAPI app
# app = FastAPI()
#
# # Load the question-answering pipeline once
# qa_pipeline = tf.pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
# context = """
# Welcome to our service! You can manage your account, reset your password, view your profile, and more.
# If you need help, our support team is available 24/7. We aim to assist with all your questions regarding our services.
#
# If you're saying hello, here's our greeting: Hello! How can we assist you today?
# For password resets, visit the settings page. For account details, visit your profile.
# Feel free to ask anything else!
# """
#
# # Function to detect greetings
# def detect_greeting(question: str):
#     greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
#     return any(greeting in question.lower() for greeting in greetings)
# # Define the request body model
# class Question(BaseModel):
#     question: str
#
# @app.post("/get-answer/")
# async def get_answer(question: Question):
#     # Check if the question is a greeting
#     if detect_greeting(question.question):
#         return {"answer": "Hello! How can I assist you today?"}
#     result = qa_pipeline(question=question.question, context=context)
#
#     # Check for low confidence (adjust the threshold as needed)
#     if result['score'] < 0.5:  # Example threshold
#         raise HTTPException(status_code=400, detail="I'm not equipped to answer that, but feel free to ask about our services!")
#
#     return {"answer": result['answer']}
#
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import torch

# Initialize FastAPI app
app = FastAPI()

# Load the SentenceTransformer model for embedding generation
embedder = SentenceTransformer('all-mpnet-base-v2')

# Sample knowledge base (replace this with your actual knowledge base)
knowledge_base = [
    "GDPR is a regulation that protects personal data.",
    "Data subjects have the right to access their personal data.",
    "Companies must obtain consent for processing personal data.",
    "Data retention policies must comply with GDPR.",
    "The right to be forgotten is a key principle of GDPR.",
    "Hello! How can I assist you today?",
    "Hi there! Feel free to ask any questions you have.",
    "Goodbye! Have a great day ahead.",
    "Thank you for reaching out. How can I help?"
]


# Embed the knowledge base
corpus_embeddings = embedder.encode(knowledge_base, convert_to_tensor=True)

# Load distilgpt2 model
model_name = 'distilgpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)


# Define the request model for user queries
class QueryRequest(BaseModel):
    query: str
    top_k: int = 2  # Default to 2 top relevant documents


# @app.post("/generate-response/")
# async def generate_response_with_rag(request: QueryRequest):
#     """
#     API endpoint that generates a response using RAG (Retrieval Augmented Generation),
#     with improved handling of context and query.
#     """
#     query = request.query.lower()  # Convert query to lowercase for easier matching
#     top_k = request.top_k
#
#     # Check for greetings and extract the remaining part of the query
#     greeting_response = ""
#     if any(greet in query for greet in ["hello", "hi", "hey", "good evening"]):
#         greeting_response = "Hello! "
#
#     if any(farewell in query for farewell in ["bye", "goodbye", "see you"]):
#         return {"response": "Goodbye! Have a great day ahead."}
#
#     if "thank you" in query:
#         return {"response": "You're welcome! Feel free to ask if you need further assistance."}
#
#     # If there’s a greeting, split the greeting from the question
#     for greet in ["hello", "hi", "hey", "good evening"]:
#         if greet in query:
#             query = query.replace(greet, "").strip()  # Remove the greeting from the query
#
#     try:
#         if query:  # If there’s still a question after greeting
#             # Embed the remaining query
#             query_embedding = embedder.encode(query, convert_to_tensor=True)
#
#             # Calculate cosine similarity between the query and the knowledge base
#             cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
#
#             # Sort the scores in descending order and get the top_k most relevant documents
#             top_results = torch.topk(cos_scores, k=top_k)
#
#             # Create a context string from the relevant documents
#             context = ""
#             for score, idx in zip(top_results[0], top_results[1]):
#                 context += knowledge_base[idx] + " "  # Merge relevant context sentences
#
#             # Generate a response using distilgpt2 with the context as background
#             input_text = f"{context}\nQuery: {query}\n"
#             input_ids = tokenizer.encode(input_text, return_tensors='pt')
#             output = model.generate(input_ids, max_length=100, do_sample=True, pad_token_id=tokenizer.eos_token_id)
#
#             # Get the generated response
#             response = tokenizer.decode(output[0], skip_special_tokens=True)
#
#             # Combine greeting with generated response if applicable
#             return {"response": greeting_response + response.strip()}
#
#         else:  # If only a greeting is provided
#             return {"response": greeting_response.strip()}
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# Run the FastAPI app using Uvicorn

@app.post("/generate-response/")
async def generate_response_with_rag(request: QueryRequest):
    """
    API endpoint that generates a response using RAG (Retrieval Augmented Generation),
    with improved handling of context and query.
    """
    query = request.query.lower()  # Convert query to lowercase for easier matching
    top_k = request.top_k

    # Check for greetings and extract the remaining part of the query
    greeting_response = ""
    if any(greet in query for greet in ["hello", "hi", "hey", "good evening"]):
        greeting_response = "Hello! "

    if any(farewell in query for farewell in ["bye", "goodbye", "see you"]):
        return {"response": "Goodbye! Have a great day ahead."}

    if "thank you" in query:
        return {"response": "You're welcome! Feel free to ask if you need further assistance."}

    # If there’s a greeting, split the greeting from the question
    for greet in ["hello", "hi", "hey", "good evening"]:
        if greet in query:
            query = query.replace(greet, "").strip()  # Remove the greeting from the query

    try:
        if query:  # If there’s still a question after greeting
            # Embed the remaining query
            query_embedding = embedder.encode(query, convert_to_tensor=True)

            # Calculate cosine similarity between the query and the knowledge base
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

            # Sort the scores in descending order and get the top_k most relevant documents
            top_results = torch.topk(cos_scores, k=top_k)

            # Create a context string from the relevant documents
            context = ""
            for score, idx in zip(top_results[0], top_results[1]):
                context += knowledge_base[idx] + " "  # Merge relevant context sentences

            # Generate a response using distilgpt2 with the context as background
            # This time, keep the prompt simple: context + query, no "Query:" or other labels
            input_text = f"{context}"
            input_ids = tokenizer.encode(input_text, return_tensors='pt')
            output = model.generate(input_ids, max_length=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)

            # Get the generated response
            response = tokenizer.decode(output[0], skip_special_tokens=True)

            # Combine greeting with generated response if applicable
            return {"response": greeting_response + response.strip()}

        else:  # If only a greeting is provided
            return {"response": greeting_response.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5583)
