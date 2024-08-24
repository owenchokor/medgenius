from medgenius.download_and_upload import copy_pdfs_from_local, download_pdfs_from_s3, create_and_save_bedrock_index
import argparse

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