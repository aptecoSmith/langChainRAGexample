from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma


from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.llms import Ollama
from retrieve_docs import return_markdown_as_texts
import os
import logging
import textwrap

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#ollama pull gdisney/zephyr-uncensored

#llm = Ollama(model="llama2")
llm = Ollama(model="zephyr")
#check ollama is working
# response = llm("The first man on the moon was ... think step by step")
#
# print(response)


# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# data = loader.load()
#
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# all_splits = text_splitter.split_documents(data)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Move up two levels to get the project root, excluding .venv
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
directory_to_search = 'markdowns'
target_dir = os.path.join(ROOT_DIR,directory_to_search)
all_splits = return_markdown_as_texts(target_dir)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})
retrieved_docs = retriever.invoke("What are available channels in orbit?")

print('*************Vector store matches**************')
print(retrieved_docs[0].page_content)
print('*************Vector store matches ends**************')


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

import time
from langchain import hub
from langchain.chains import RetrievalQA
from prompts import rag_prompt_custom
#overriden later but still useful
prompt = hub.pull("rlm/rag-prompt-llama")#grab a correct system prompt

prompt = rag_prompt_custom # use your own - if using zephyr you need to use your own for now
#override sys prompt

#directly set the text only of the system prompt
#prompt.messages[0].prompt.template = ''

# RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

from terminal_outputs import ColorfulPrinterColorama

printer = ColorfulPrinterColorama()
# Start timer
start_time = time.time()
question = "What are the available channels in orbit?"
print('------------------------------------------- Agent ---------------------------')
print("\n")
printer.question_print(question)
result = qa_chain({"query": question})
response= result["result"]
printer.answer_print(response)
# End timer
end_time = time.time()
# Calculate and print elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")



