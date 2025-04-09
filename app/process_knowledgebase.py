from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
from langchain.schema import Document
import re
from typing import List, Dict
import glob
from tqdm import tqdm


def extract_metadata(md_content: str) -> Dict:
    """
        Extracts created date, tags, and author from Markdown frontmatter
        returns Dict Datatype.
    """
    metadata = {}
    if "Created on:" in md_content:
        metadata["created_date"] = re.search(r"Created on: (.+?)\n", md_content).group(1)
    if "Tags:" in md_content:
        tags = re.search(r"Tags: (.+?)\n", md_content).group(1)
        metadata["tags"] = tags.replace("|", ",")
    return metadata

def split_special_blocks(content: str) -> List[str]:
    """
        Split sections like 'Eligibility Criteria' into standalone chunks
    """
    blocks = []
    current_block = []
    
    for line in content.split("\n"):
        if line.startswith("**") and line.endswith("**"):
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
        current_block.append(line)
    
    if current_block:
        blocks.append("\n".join(current_block))
    return blocks

def process_immigration_doc(file_path: str) -> List[Document]:
    """
        extracts metadata, splits by headers and then further processess each section
        returns a list of Document
    """    
    with open(file_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    
    # 1. Extract metadata
    metadata = extract_metadata(md_content)
    metadata["source"] = file_path
    
    # 2. Split by headers (keep H2 and H3 as units)
    headers_to_split_on = [("##", "Section"), ("###", "Subsection")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    header_splits = markdown_splitter.split_text(md_content)
    
    # 3. Further process each section
    final_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    for doc in header_splits:
        # Special handling for key sections
        if any(k in doc.page_content for k in ["Eligibility", "Required Documents", "Processing Time"]):
            sub_blocks = split_special_blocks(doc.page_content)
            for block in sub_blocks:
                sub_chunks = text_splitter.split_text(block)
                for chunk in sub_chunks:
                    new_doc = Document(
                        page_content=chunk,
                        metadata={**doc.metadata, **metadata}
                    )
                    final_chunks.append(new_doc)
        else:
            chunks = text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                new_doc = Document(
                    page_content=chunk,
                    metadata={**doc.metadata, **metadata}
                )
                final_chunks.append(new_doc)
    
    return final_chunks

if __name__ == "__main__":
    md_files = glob.glob("../knowledge_base/*.md")
    all_chunks = []
    
    for file_path in tqdm(md_files):
        all_chunks.extend(process_immigration_doc(file_path))   

    print(f"Processed {len(all_chunks)} chunks from {len(md_files)} files")
