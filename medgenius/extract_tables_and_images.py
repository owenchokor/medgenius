import camelot
import pandas as pd
import fitz 
import io
from io import BytesIO
from PIL import Image
import base64
import json
import boto3
import os
import glob
import yaml

os.environ["PATH"] += os.pathsep + os.path.abspath('./util/gs10031w64.exe')
# AWS Bedrock client setup
runtime = boto3.client("bedrock-runtime", "us-west-2")



def extract_tables(pdf_path, page_number):
    """
    PDF의 특정 페이지에서 테이블을 추출하는 함수

    :param pdf_path: PDF 파일 경로
    :param page_number: 테이블을 추출할 페이지 번호 (1부터 시작)
    :return: 추출된 테이블 데이터 리스트 (각 테이블은 DataFrame 형태)
    """
    # Camelot을 사용하여 특정 페이지에서 테이블 추출
    tables = camelot.read_pdf(pdf_path, pages=str(page_number), strip_text='\n')
    
    # 각 테이블을 string 형식으로 변환하여 리스트로 저장
    table_dataframes = [table.df.apply(lambda x: x.ffill() if x.name != 0 else x) for table in tables]
    
    result = list(map(lambda x : _analyze_table(x), table_dataframes))
    
    return result



def extract_images(pdf_path, page_number):
    pdf_document = fitz.open(pdf_path)
    page = pdf_document[page_number]
    image_list = page.get_images()
    result = []

    # 추출된 각 이미지에 대해
    for img_index, img in enumerate(image_list):
        # 이미지 가져오기
        xref = img[0]
        base_image = pdf_document.extract_image(xref)
        image_bytes = base_image["image"]

        # 이미지 포맷 결정
        ext = base_image["ext"]

        # 이미지 객체 생성
        image = Image.open(io.BytesIO(image_bytes))
        image_result_text = f'[Description Image #{img_index} of this page] ' + _analyze_image(image, 'image_in_a_sentence')
        result.append(image_result_text)
    return result


def _load_prompts(file_path='./medgenius/prompts.yaml'):
    with open(file_path, 'r') as file:
        prompts = yaml.safe_load(file)
    return prompts

def _analyze_image(image_pil, query_type):
    try:
        buffered = BytesIO()
        image_pil.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()

        # 이미지 base64로 인코딩
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        print(f'passing image because of {e}')
        return ''

    # 요청 바디 생성
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": encoded_image,
                            },
                        },
                        {"type": "text", "text": _load_prompts()[query_type]},
                    ],
                }
            ],
        }
    )

    # 모델에 요청 보내기
    response = runtime.invoke_model(
        modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
        body=body
    )

    # 응답 처리
    response_body = json.loads(response.get("body").read())
    result_text = response_body['content'][0]['text']

    return result_text

def _analyze_table(table_pandas, query_type='tabular'):
    result = "read this as dataframe: {"
    for index, row in table_pandas.iterrows():
        result += f"{index}:"+"{{"
        for col in table_pandas.columns:
            result += f"{col}:{{{row[col]}}}, "
        result = result.rstrip(', ')  # 마지막 콤마 제거
        result += "}, "
    result = result.rstrip(', ')  # 마지막 콤마 제거
    result += "}"
    return result