import os

from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def find_md_files(directory_path):
    md_files = [os.path.join(dp, f) for dp, _, filenames in os.walk(directory_path)
                for f in filenames if f.lower().endswith('.md')]
    return md_files

def return_markdown_as_texts(directory_to_search):

    # Example usage:

    markdown_files = find_md_files(directory_to_search)
    print(markdown_files)
    # load the docs
    loader = DirectoryLoader(directory_to_search, glob="**/*.md",
                             show_progress=True, use_multithreading=True)
    documents = loader.load()
    print('Docs loaded')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts