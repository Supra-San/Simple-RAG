import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma 
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import gradio as gr 

# Load the OpenAI API Key from an external file for security
with open("config/api_key.txt", "r") as file:
    OPENAI_API_KEY = file.read().strip()

# Define the function to generate answers
def answer_question(file, question, temperature, max_tokens):
    """
    Processes the uploaded PDF and answers the user's question.

    Parameters:
        file (file): Uploaded PDF file.
        question (str): User's question.
        temperature (float): LLM temperature setting.
        max_tokens (int): Maximum tokens for the LLM response.

    Returns:
        str: Generated answer or error message.
    """
    if file is None:
        return "Please upload a PDF file."
    if not question.strip():
        return "Please enter a question."

    try:
        # Load the uploaded PDF file
        loader = PyPDFLoader(file.name)
        documents = loader.load()

        # Split the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)

        # Initialize the embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Initialize the vectorstore with the new document
        vectordb = Chroma.from_documents(texts, embeddings)

        # Update the retriever
        retriever = vectordb.as_retriever()

        # Update the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(
                openai_api_key=OPENAI_API_KEY,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
        )

        # Ask the user's question
        result = qa_chain({"query": question})

        answer = result['result']
        source = result['source_documents'][0].page_content[:500] if result['source_documents'] else "No source found."

        return f"{answer}"

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create the Gradio app
with gr.Blocks(theme="compact") as app:
    gr.Markdown("""
    # RAG Simple Application
    Upload a PDF document and ask any question related to its content. This app uses AI to retrieve and answer your queries!
    """)

    with gr.Row():
        with gr.Column(scale=2, min_width=400):
            gr.Markdown("### Step 1: Upload Your Document")
            file_input = gr.File(label="Upload PDF File", file_types=[".pdf"])

        with gr.Column(scale=2, min_width=400):
            gr.Markdown("### Step 2: Ask a Question")
            question_input = gr.Textbox(label="Ask a Question", placeholder="Type your question here...", lines=9)

        with gr.Column():
            gr.Markdown("### Step 3: Adjust Model Parameters(Optional)")
            temperature_slider = gr.Slider(label="Temperature", minimum=0, maximum=1, value=0.7, step=0.1)
            token_slider = gr.Slider(label="Max Tokens", minimum=500, maximum=2000, value=1000, step=100)
            

    with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.Button("Clear", variant="secondary")

    output = gr.Textbox(label="Answer", placeholder="The answer will appear here...", lines=2)

    # Link submit button to the answer function
    submit_btn.click(fn=answer_question, inputs=[file_input, question_input, temperature_slider, token_slider], outputs=output)

    # Define the clear function
    def clear_inputs():
        return None, "", 0.7, 1000, ""

    clear_btn.click(fn=clear_inputs, inputs=[], outputs=[file_input, question_input, temperature_slider, token_slider, output])

    gr.Markdown("""
    **Tips:**
    - Ensure the uploaded PDF is clear and well-formatted.
    - Adjust the temperature and token settings for better results.
    """)

# Run the app
app.launch()
