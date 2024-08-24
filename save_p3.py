import os
import boto3
import fitz  # PyMuPDF
import argparse
from tqdm import tqdm
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.docstore.document import Document
from preprocess import PDFProcessor

# S3에서 PDF 파일을 다운로드하는 함수
def download_pdfs_from_s3(bucket_name, prefix, download_path='/tmp/pdfs'):
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    pdf_files = []
    total_files = len(response.get('Contents', []))
    
    for obj in tqdm(response.get('Contents', []), desc="Downloading PDFs", total=total_files):
        if obj['Key'].endswith('.pdf'):
            try:
                file_name = os.path.join(download_path, obj['Key'].split('/')[-1])
                s3_client.download_file(bucket_name, obj['Key'], file_name)
                pdf_files.append(file_name)
            except OSError:
                continue
    
    return pdf_files

# 로컬 폴더에서 pdf 가져오는 함수
def copy_pdfs_from_local(prefix, local_path, download_path='/tmp/pdfs'):
    # PDF 파일들을 저장할 경로가 존재하지 않는다면 생성
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    pdf_files = []
    
    # 주어진 경로와 prefix를 기반으로 파일 목록을 가져옴
    files_to_copy = [f for f in os.listdir(local_path) if f.startswith(prefix) and f.endswith('.pdf')]
    
    total_files = len(files_to_copy)
    
    # PDF 파일을 다운로드 경로로 복사
    for file_name in tqdm(files_to_copy, desc="Copying PDFs", total=total_files):
        try:
            source_file = os.path.join(local_path, file_name)
            destination_file = os.path.join(download_path, file_name)
            shutil.copy(source_file, destination_file)
            pdf_files.append(destination_file)
        except OSError:
            continue
    
    return pdf_files

# PyMuPDF를 사용하여 PDF에서 텍스트 추출하는 함수 (오류 처리 추가)
def _extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""
    

    
# Bedrock 임베딩을 사용하여 인덱스 생성 및 벡터 DB 저장
def create_and_save_bedrock_index(pdf_paths, bucket_name, preprocess = True, vector_db_s3_path='vectorDB/'):
    # 문서 로드 및 텍스트 추출
    if preprocess:
        documents = []
        processor = PDFProcessor(pdf_paths[0])
        
        for path in tqdm(pdf_paths, desc="Processing PDFs"):
            processor.get_next(path)
            try:
                documents.append(processor.hr_index())
            except Exception as e:
                print(f"passing {path} due to {e}")
            
        return documents
    else:
        documents = []
        for path in tqdm(pdf_paths, desc="Processing PDFs"):
            text = _extract_text_from_pdf(path)
            if text:  # 텍스트가 추출된 경우에만 추가
                documents.append(Document(page_content=text))
        
        # 텍스트 분할기 정의
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
        )
        
        # Bedrock 임베딩 사용
        embeddings = BedrockEmbeddings(
            credentials_profile_name='default',
            region_name='us-east-1',
            model_id='amazon.titan-embed-text-v1'
        )
        
        # FAISS 벡터 저장소 생성
        vector_index = FAISS.from_documents(documents, embeddings)
        
    # 로컬에 저장할 경로
    local_index_path = '/tmp/vectordb_index'
    if not os.path.exists(local_index_path):
        os.makedirs(local_index_path)
    
    index_file = os.path.join(local_index_path, 'faiss_index')
    vector_index.save_local(local_index_path)
    
    # 로컬 저장 진행상황 출력
    print("Saving vector DB to local...")
    for file_name in tqdm(os.listdir(local_index_path), desc="Saving to local"):
        pass  # tqdm으로 진행 상황만 표시
    
    # S3에 저장
    s3_client = boto3.client('s3')
    print("Uploading vector DB to S3...")
    for file_name in tqdm(os.listdir(local_index_path), desc="Uploading to S3"):
        s3_client.upload_file(
            os.path.join(local_index_path, file_name), 
            bucket_name, 
            os.path.join(vector_db_s3_path, file_name)
        )
    
    print(f"Vector DB files uploaded to s3://{bucket_name}/{vector_db_s3_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--bucket_name', type=str, default='snuh-data-team2', help='Name of the S3 bucket')
    parser.add_argument('--pdf_source', type=str, choices=['local', 'download'], default='local', help='Source of the PDFs')
    parser.add_argument('--preprocess', type=bool, default=False, help='Flag to determine if preprocessing should be done (True or False)')

    args = parser.parse_args()

    if args.pdf_source == 'local':
        pdf_paths = copy_pdfs_from_local('', './data')
    elif args.pdf_source == 'download':
        pdf_paths = download_pdfs_from_s3(args.bucket_name, 'data/')

    print("S3 -> Backend Transport Done.")
    
    # 벡터 DB 생성 및 S3에 저장
    create_and_save_bedrock_index(pdf_paths, args.bucket_name, preprocess=args.preprocess)