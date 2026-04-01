#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -r ../../requirements.txt')


# In[2]:


def init_db(db_path="./chroma_db"):
    dbclient = chromadb.PersistentClient(path=db_path)
    try:
        dbclient.delete_collection(name="rag_collection")
    except:
        pass 

    collection = dbclient.create_collection(name="rag_collection")
    return dbclient, collection


# In[3]:


import os  # os를 가져와 파일 시스템 접근, 환경 변수 읽을 수 있음
from openai import OpenAI  # OpenAI의 api 사용 가능
import chromadb  # chromadb 라이브러리 쓸 수 있게 해줌
from chromadb.config import (
    Settings,
)  # Settings 클래스는 DB의 구성 옵션을 설정하는데 사용
from dotenv import load_dotenv  # 환경 변수를 로드하기 위함


# 1. 환경 변수 Load해서 api_key 가져오고 OpenAI 클라이언트(객체) 초기화

# In[4]:


load_dotenv() 
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# 2. DB 초기화 함수 (매 실행 시 DB 폴더를 삭제 후 새로 생성)

# In[5]:


# DB 초기화 함수
def init_db(db_path="./chroma_db"): # 현재 디렉토리 내의 chroma_db폴더 설정
    dbclient = chromadb.PersistentClient(path=db_path) # 지정 경로로 향하는 dbClient 생성
    # rag_collection이라는 데이터 컬렉션(모음집) 만듦
    # get_or_create옵션은 만약 해당 이름 컬렉션이 이미 존재하면 기존 컬렉션 쓰고 아님 만들고
    collection = dbclient.create_collection(name="rag_collection", get_or_create=True)
    return dbclient, collection


# 3. 텍스트 로딩 함수

# In[6]:


# 텍스트 로딩 함수 
def load_text_files(folder_path):
    docs=[] # 텍스트 파일의 (파일명, 내용) 튜플들을 저장할 빈 리스트 생성
    for filename in os.listdir(folder_path): 
		    # 현재 폴더 경로와 파일 이름을 결합하여 파일의 전체 경로를 생성
        file_path = os.path.join(folder_path, filename) 
        if file_path.endswith(".txt"): # .txt 파일들을 처리하게 함
            with open(file_path, "r", encoding="utf-8") as f: # 읽기모드로 열고 인코딩
                text = f.read() # 파일 전체 내용을 읽어서 문자열로 저장
                docs.append((filename, text)) # (파일명, 내용) 튜플을 docs 리스트에 추가
    return docs # 모든 텍스트 파일의 (파일명, 내용) 튜플 리스트를 반환


# 4. 주어진 text를 임베딩 벡터로 변환하는 함수

# In[7]:


# 주어진 text를 임베딩 벡터로 변환하는 함수 
def get_embedding(text, model="text-embedding-3-large"):
		# 여기서 client는 앞서 초기화한 OpenAI 클라이언트
    response = client.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding # 응답 객체의 data 리스트에서 embedding 필드 추출
    return embedding 


# 5. 원천 데이터 청크 단위로 나누고 overlap 사이즈 조절하는 함수

# In[8]:


# 원천 데이터 청크 단위로 나누고 overlap 사이즈 조절하는 함수
def chunk_text(text, chunk_size=400, chunk_overlap=50):
    chunks = [] # 분할된 텍스트 청크들을 저장할 리스트
    start = 0 # 청크를 시작할 위치를 나타내는 인덱스
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end] # 텍스트에서 start부터 end까지 부분 문자열을 추출
        chunks.append(chunk) # 추출한 청크를 리스트에 추가
        start = end - chunk_overlap # overlap 적용

        if start < 0: # 음수가 될 수 있으니 예외 처리
            start = 0

        if start >= len(text): # 종료 시그널
            break

    return chunks # 모든 청크가 저장된 리스트를 반환


# 6. 문서로드 -> 청크 나누고 -> 임베딩 생성 후 DB 삽입

# In[9]:


# 문서로드 -> 청크 나누고 -> 임베딩 생성 후 DB 삽입
if __name__ == "__main__":
    # db 초기화
    dbclient, collection = init_db("./chroma_db")

    folder_path = "./source_data" # 데이터 가져다 쓸 경로 지정
    docs = load_text_files(folder_path) # 처리할 문서 데이터 메모리로 불러오기

    doc_id = 0
    for filename, text in docs: 
        chunks = chunk_text(text, chunk_size=400, chunk_overlap=50) # chunking
        for idx, chunk in enumerate(chunks): # 각 청크와 해당 청크의 인덱스 가져옴
            doc_id += 1 # 인덱스 하나씩 증가 시키면서
            embedding = get_embedding(chunk) # 각 청크 임베딩 벡터 생성
            # vectorDB에 다음 정보 추가
            collection.add(
                documents=[chunk], # 실제 청크 text
                embeddings=[embedding], # 생성된 임베딩 벡터
                metadatas=[{"filename": filename, "chunk_index": idx}], # 파일 이름과 청크 인덱스를 포함하는 메타데이터
                ids=[str(doc_id)] # 각 청크의 Unique한 id 저장
                # 이 고유 id를 통해 db에서 업데이트, 삭제등의 작업 가능 
            )

    print("모든 문서 벡터DB에 저장 완료")

