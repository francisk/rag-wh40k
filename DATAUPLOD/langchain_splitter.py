from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import json

def process_markdown_with_langchain(size, over_lap, text: str,output_file:str) -> list:
    """
    使用langchain的MarkdownHeaderTextSplitter 
    首先通过markdown的标题进行分割
    然后使用RecursiveCharacterTextSplitter进行分割
    """
    # 首先使用 MarkdownHeaderTextSplitter 按标题分割
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###","Header 3")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_splits = markdown_splitter.split_text(text)
    
    # 然后使用 RecursiveCharacterTextSplitter 进行进一步分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,  # 每个块的大小
        chunk_overlap=over_lap,  # 块之间的重叠部分
        length_function=len,
        separators=["\n\n", "\n", " ", "----"]  # 分割符优先级
    )
    
    # 对每个标题分割后的文本进行进一步分割
    final_chunks = []
    for split in md_splits:
        # 获取标题信息
        header_info = {k: v for k, v in split.metadata.items() if k.startswith('Header')}
        header_text = " ".join(header_info.values())
        
        # 分割文本
        sub_chunks = text_splitter.split_text(split.page_content)
        
        # 将标题信息添加到每个子块
        for chunk in sub_chunks:
            if header_text:
                final_chunks.append(f"{header_text}\n{chunk}")
            else:
                final_chunks.append(chunk)
    
    recursive_to_txt(final_chunks,output_file)
    return final_chunks

def recursive_to_txt(chunks: list, output_file: str):
    """
    将分割后的文本块保存为 TXT 文件，使用 ----{chunk number}---- 作为分隔符
    
    Args:
        chunks (list): 文本块列表
        output_file (str): 输出文件名
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"总块数: {len(chunks)}\n\n")
        for i, chunk in enumerate(chunks, 1):
            f.write(f"----{i}----\n")
            f.write(chunk.strip())
            f.write("\n\n")

if __name__ == "__main__":
    source_file = '40kcorerule.md'
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()
        process_markdown_with_langchain(2000, 150, content,"result_1.txt") 