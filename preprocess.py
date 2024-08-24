from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from extract_tables_and_images import extract_images, extract_tables
from tqdm import tqdm

class PDFProcessor():
    def __init__(self, file_path):
        self.file_path = file_path
    def get_next(self, file_path):
        self.file_path = file_path

    def hr_index(self):
        # 2. 데이터 소스 정의 및 PDFLoader로 데이터 로드
        data_load = PyPDFLoader(self.file_path).load()
        
        # 3. 벡터 DB 생성 준비
        data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=100, chunk_overlap=10)
        data_embeddings = BedrockEmbeddings(
            credentials_profile_name='default',
            region_name='us-east-1',
            model_id='amazon.titan-embed-text-v1'#'cohere.command-r-plus-v1:0'
        )
        
        db_index = None
        
        # 4. 각 페이지별로 데이터 처리
        for page_num, page in tqdm(enumerate(data_load), desc='vectorizing', total=len(data_load)):
            # 페이지별로 텍스트, 테이블, 이미지 데이터 추출
            page_text = page.page_content
            page_tables = extract_tables(self.file_path, page_number=page_num)
            page_images = extract_images(self.file_path, page_number=page_num)
            
            # 텍스트 데이터 벡터화 및 인덱스 추가
            if page_text.strip():
                text_documents = data_split.split_text(page_text)
                text_index = FAISS.from_texts(text_documents, data_embeddings)
                if db_index is None:
                    db_index = text_index
                else:
                    db_index.merge_from(text_index)
            
            # 테이블 데이터 벡터화 및 인덱스 추가
            if page_tables:
                for table in page_tables:
                    table_text = table  # assuming `extract_tables` returns text
                    table_documents = data_split.split_text(table_text)
                    table_index = FAISS.from_texts(table_documents, data_embeddings)
                    db_index.merge_from(table_index)
            
            # 이미지 데이터 벡터화 및 인덱스 추가
            if page_images:
                for image in page_images:
                    image_text = image  # assuming `extract_images` returns text
                    image_documents = data_split.split_text(image_text)
                    image_index = FAISS.from_texts(image_documents, data_embeddings)
                    db_index.merge_from(image_index)
        
        return db_index