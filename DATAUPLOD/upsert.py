import asyncio
from pinecone import Pinecone, Vector
from langchain_splitter import process_markdown_with_langchain, recursive_to_txt
from langchain_community.embeddings import OpenAIEmbeddings
import os
from datetime import datetime
import argparse

# Pinecone 配置
PINECONE_API_KEY = ""
PINECONE_INDEX = "wh40kcodex"
# OpenAI 配置
OPENAI_API_KEY = ""
async def upsert_to_pinecone(texts: list, faction: str, source_file: str = "content.md"):
    # 初始化 OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # 初始化 Pinecone 客户端
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # 获取当前时间
    current_time = datetime.now().isoformat()
    
    async with pc.IndexAsyncio(PINECONE_INDEX) as idx:
        # 准备向量数据
        vectors = []
        for i, text in enumerate(texts):
            # 获取文本的向量表示
            embedding = await embeddings.aembed_query(text)           
            # 构建丰富的 metadata
            metadata = {
                "text": text,
                "source_file": source_file,
                "chunk_index": i,
                "faction": faction,
                "total_chunks": len(texts),
                "timestamp": current_time,
                "content_type": "markdown",
                "chunk_size": len(text),
                "language": "zh"  # 假设是中文内容
            }
            
            vectors.append(Vector(
                id=f"doc_{source_file}_{i}_{current_time}",
                values=embedding,
                metadata=metadata
            ))
        
        # 上传向量
        await idx.upsert(vectors=vectors)
        print(f"成功上传 {len(vectors)} 个文档到 Pinecone！")
        
        # 查看索引统计信息
        stats = await idx.describe_index_stats()
        print("索引统计信息：", stats)

async def main(source_file: str, faction: str, chunk_size: int = 1500, chunk_overlap: int = 150, output_file: str = "result.txt"):
    """
    主函数
    
    Args:
        source_file (str): 源文件路径
        faction (str): 派系名称
        chunk_size (int): 文本块大小
        chunk_overlap (int): 文本块重叠大小
        output_file (str): 输出文件名
    """
    # 读取文档内容
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用 LangChain 分割文本
    chunks = process_markdown_with_langchain(chunk_size, chunk_overlap, content, output_file)
    print(f"文本已分割为 {len(chunks)} 个块")
    
    # 保存分割结果到文件
    recursive_to_txt(chunks, output_file)
    
    # 上传到 Pinecone
    await upsert_to_pinecone(chunks, faction, source_file)

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='处理文档并上传到 Pinecone')
    parser.add_argument('--source', type=str, default='content.md', help='源文件路径')
    parser.add_argument('--faction', type=str, required=True, help='派系名称')
    parser.add_argument('--chunk-size', type=int, default=1500, help='文本块大小')
    parser.add_argument('--chunk-overlap', type=int, default=150, help='文本块重叠大小')
    parser.add_argument('--output', type=str, default='result.txt', help='输出文件名')
    
    args = parser.parse_args()
    
    # 运行主函数
    asyncio.run(main(
        source_file=args.source,
        faction=args.faction,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        output_file=args.output
    ))
