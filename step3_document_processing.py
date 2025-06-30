"""
3단계: 문서 처리 및 임베딩 생성
이 스크립트는 PDF 문서를 Claude 3.7 Sonnet으로 파싱하고 Titan2 임베딩을 생성하여 OpenSearch에 인덱싱합니다.
refer_parser_prompt.md 프롬프트를 사용하여 구조화된 마크다운으로 변환합니다.
"""

import json
import boto3
import time
import base64
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import uuid
from datetime import datetime
import re

class DocumentProcessor:
    def __init__(self, config_file='opensearch_config.json'):
        """
        문서 처리기 초기화
        
        Args:
            config_file (str): OpenSearch 설정 파일 경로
        """
        # 설정 로드
        self.config = self.load_config(config_file)
        if not self.config:
            raise Exception("설정 파일을 로드할 수 없습니다")
        
        # AWS 클라이언트 초기화
        self.region = self.config['region']
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.region)
        self.s3_client = boto3.client('s3', region_name=self.region)
        
        # S3 버킷 이름 생성 (고유한 이름)
        account_id = boto3.client('sts', region_name=self.region).get_caller_identity()['Account']
        timestamp = str(int(time.time()))[-6:]
        self.bucket_name = f"rag-car-manual-{account_id}"
        
        # Claude API 호출 제한 관리
        self.claude_call_count = 0  # 현재 호출 횟수
        self.claude_call_limit = 5  # 1분당 최대 호출 횟수
        self.claude_window_start = time.time()  # 현재 윈도우 시작 시간
        self.claude_window_duration = 60  # 윈도우 지속 시간 (초)
        
        # OpenSearch 클라이언트 초기화
        credentials = boto3.Session().get_credentials()
        self.awsauth = AWSV4SignerAuth(credentials, self.region, "aoss")
        self.oss_client = OpenSearch(
            hosts=[{"host": self.config['host'], "port": 443}],
            http_auth=self.awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300,
        )
        
        # 파서 프롬프트 로드
        self.parser_prompt = self.load_parser_prompt()
        
        print(f"🔗 문서 처리기 초기화 완료")
        print(f"🌍 리전: {self.region}")
        print(f"🪣 S3 버킷: {self.bucket_name}")
        print(f"🔗 OpenSearch 엔드포인트: {self.config['endpoint']}")
        print(f"📄 인덱스: {self.config['index_name']}")
        print(f"🤖 LLM: Claude 3.7 Sonnet")
        print(f"🧠 임베딩: {self.config['embedding_model']}")
    
    def create_s3_bucket(self):
        """
        S3 버킷 생성 (이미 존재하면 그대로 사용)
        
        Returns:
            bool: 성공 여부
        """
        try:
            # 버킷 존재 확인
            try:
                self.s3_client.head_bucket(Bucket=self.bucket_name)
                print(f"✅ S3 버킷이 이미 존재합니다: {self.bucket_name}")
                return True
            except Exception as e:
                # 404 또는 NoSuchBucket 오류는 버킷이 없다는 의미이므로 생성 진행
                if "404" in str(e) or "NoSuchBucket" in str(e) or "Not Found" in str(e):
                    print(f"📝 S3 버킷이 존재하지 않습니다. 새로 생성합니다: {self.bucket_name}")
                else:
                    # 다른 오류는 재발생
                    raise e
            
            # 버킷 생성
            print(f"🪣 S3 버킷 생성 중: {self.bucket_name}")
            
            if self.region == 'us-east-1':
                # us-east-1은 LocationConstraint를 지정하지 않음
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                # 다른 리전은 LocationConstraint 필요
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            
            print(f"✅ S3 버킷 생성 완료: {self.bucket_name}")
            return True
            
        except Exception as e:
            print(f"❌ S3 버킷 생성 실패: {e}")
            return False
    
    def upload_to_s3(self, file_path: str) -> str:
        """
        파일을 S3에 업로드
        
        Args:
            file_path (str): 업로드할 파일 경로
            
        Returns:
            str: S3 URI
        """
        try:
            file_name = Path(file_path).name
            s3_key = f"documents/{file_name}"
            
            print(f"📤 S3에 파일 업로드 중: {file_name}")
            
            # 파일 업로드
            self.s3_client.upload_file(
                file_path, 
                self.bucket_name, 
                s3_key,
                ExtraArgs={'ContentType': 'application/pdf'}
            )
            
            # S3 URI 생성
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            print(f"✅ S3 업로드 완료: {s3_uri}")
            
            return s3_uri
            
        except Exception as e:
            print(f"❌ S3 업로드 오류: {e}")
            return None
    
    def save_page_image_to_s3(self, image_base64: str, page_number: int, pdf_filename: str) -> str:
        """
        페이지 이미지를 S3에 저장
        
        Args:
            image_base64 (str): base64 인코딩된 이미지
            page_number (int): 페이지 번호 (0부터 시작)
            pdf_filename (str): PDF 파일명
            
        Returns:
            str: 이미지 S3 URI
        """
        try:
            # 파일명에서 확장자 제거
            base_filename = Path(pdf_filename).stem
            image_key = f"images/{base_filename}_page_{page_number + 1}.png"
            
            # base64 디코딩
            image_data = base64.b64decode(image_base64)
            
            # S3 업로드
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=image_key,
                Body=image_data,
                ContentType='image/png'
            )
            
            # S3 URI 생성
            image_s3_uri = f"s3://{self.bucket_name}/{image_key}"
            print(f"🖼️ 페이지 {page_number + 1} 이미지 저장: {image_s3_uri}")
            
            return image_s3_uri
            
        except Exception as e:
            print(f"❌ 이미지 S3 저장 오류: {e}")
            return None
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            print(f"✅ S3 업로드 완료: {s3_uri}")
            
            return s3_uri
            
        except Exception as e:
            print(f"❌ S3 업로드 실패: {e}")
            return ""
    
    def load_config(self, config_file):
        """설정 파일 로드"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 설정 파일 로드 실패: {e}")
            return None
    
    def load_parser_prompt(self, prompt_file='refer_parser_prompt.md'):
        """파서 프롬프트 로드"""
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt = f.read()
            print(f"📄 파서 프롬프트 로드 완료: {prompt_file}")
            return prompt
        except Exception as e:
            print(f"❌ 파서 프롬프트 로드 실패: {e}")
            return None
    
    def pdf_to_images(self, pdf_path: str) -> List[Dict]:
        """
        PDF를 페이지별 이미지로 변환
        
        Args:
            pdf_path (str): PDF 파일 경로
            
        Returns:
            List[Dict]: 페이지 정보 리스트
        """
        print(f"📄 PDF 파일 로드 중: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            pages = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # 페이지를 이미지로 변환 (300 DPI)
                mat = fitz.Matrix(300/72, 300/72)  # 300 DPI 변환 매트릭스
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Base64 인코딩
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                pages.append({
                    'page_number': page_num,
                    'image_base64': img_base64,
                    'image_format': 'png'
                })
                
                print(f"  📄 페이지 {page_num + 1}/{len(doc)} 변환 완료")
            
            doc.close()
            print(f"✅ PDF 변환 완료: {len(pages)}페이지")
            return pages
            
        except Exception as e:
            print(f"❌ PDF 변환 오류: {e}")
            return []
    
    def manage_claude_rate_limit(self):
        """
        Claude API 호출 제한 관리
        1분당 5회 제한을 준수하기 위해 필요시 대기
        """
        current_time = time.time()
        
        # 새로운 1분 윈도우가 시작되었는지 확인
        if current_time - self.claude_window_start >= self.claude_window_duration:
            # 새 윈도우 시작
            self.claude_window_start = current_time
            self.claude_call_count = 0
            print(f"🔄 새로운 API 호출 윈도우 시작")
        
        # 호출 제한에 도달했는지 확인
        if self.claude_call_count >= self.claude_call_limit:
            # 다음 윈도우까지 대기 시간 계산
            elapsed = current_time - self.claude_window_start
            wait_time = self.claude_window_duration - elapsed
            
            if wait_time > 0:
                print(f"⏳ API 호출 제한 도달 ({self.claude_call_count}/{self.claude_call_limit})")
                print(f"⏰ {wait_time:.1f}초 대기 중...")
                
                # 진행률 표시하며 대기
                for remaining in range(int(wait_time), 0, -1):
                    print(f"   ⏱️  {remaining}초 남음...", end='\r')
                    time.sleep(1)
                
                print(f"   ✅ 대기 완료!                    ")
                
                # 새 윈도우 시작
                self.claude_window_start = time.time()
                self.claude_call_count = 0
        
        # 호출 횟수 증가
        self.claude_call_count += 1
        print(f"🤖 Claude API 호출 ({self.claude_call_count}/{self.claude_call_limit})")
    
    def parse_page_with_claude(self, image_base64: str, page_number: int) -> str:
        """
        Claude 3.7 Sonnet으로 페이지 파싱 (API 호출 제한 적용)
        
        Args:
            image_base64 (str): Base64 인코딩된 이미지
            page_number (int): 페이지 번호
            
        Returns:
            str: 파싱된 마크다운 텍스트
        """
        try:
            # API 호출 제한 관리
            self.manage_claude_rate_limit()
            
            print(f"🤖 Claude로 페이지 {page_number + 1} 파싱 중...")
            
            # Claude 3.7 Sonnet 호출
            response = self.bedrock_client.invoke_model(
                modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4000,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self.parser_prompt
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": image_base64
                                    }
                                }
                            ]
                        }
                    ]
                })
            )
            
            # 응답 파싱
            result = json.loads(response['body'].read())
            print (f'{result=}')
            content = result['content'][0]['text']
            
            # <markdown></markdown> 태그에서 내용 추출
            markdown_match = re.search(r'<markdown>(.*?)</markdown>', content, re.DOTALL)
            if markdown_match:
                parsed_content = markdown_match.group(1).strip()
                print(f"✅ 페이지 {page_number + 1} 파싱 완료 ({len(parsed_content)}자)")
                return parsed_content
            else:
                print(f"⚠️ 페이지 {page_number + 1}: 마크다운 태그를 찾을 수 없음")
                return content.strip()
                
        except Exception as e:
            print(f"❌ 페이지 {page_number + 1} 파싱 실패: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        텍스트를 청크로 분할 (마크다운 구조 고려)
        
        Args:
            text (str): 분할할 텍스트
            chunk_size (int): 청크 크기
            overlap (int): 겹치는 부분 크기
            
        Returns:
            List[str]: 청크 리스트
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 청크 끝이 텍스트 끝을 넘지 않도록 조정
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # 문장 경계에서 자르기 시도
            chunk_text = text[start:end]
            
            # 마크다운 헤더나 문장 끝에서 자르기
            for delimiter in ['\n## ', '\n# ', '\n* ', '. ', '.\n', '\n']:
                last_delim = chunk_text.rfind(delimiter)
                if last_delim > chunk_size // 2:  # 청크의 절반 이상에서 발견된 경우
                    end = start + last_delim + len(delimiter)
                    break
            
            chunks.append(text[start:end])
            start = end - overlap
        
        print(f"📝 텍스트 청킹 완료: {len(chunks)}개 청크")
        return chunks
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Titan2로 임베딩 생성
        
        Args:
            text (str): 임베딩할 텍스트
            
        Returns:
            List[float]: 1024차원 벡터
        """
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.config['embedding_model'],
                body=json.dumps({
                    "inputText": text,
                    "dimensions": self.config['vector_dimension'],
                    "normalize": True
                })
            )
            
            result = json.loads(response['body'].read())
            return result['embedding']
            
        except Exception as e:
            print(f"❌ 임베딩 생성 실패: {e}")
            return []
    
    def index_document(self, chunks: List[str], source_uri: str, page_number: int, original_text: str, image_s3_uri: str = None):
        """
        문서 청크를 OpenSearch에 인덱싱
        
        Args:
            chunks (List[str]): 텍스트 청크 리스트
            source_uri (str): 소스 URI
            page_number (int): 페이지 번호
            original_text (str): 원본 텍스트
            image_s3_uri (str): 페이지 이미지 S3 URI (선택사항)
        """
        print(f"📤 페이지 {page_number + 1} 인덱싱 중... ({len(chunks)}개 청크)")
        
        for chunk_idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            try:
                # 임베딩 생성
                embedding = self.generate_embedding(chunk)
                if not embedding:
                    print(f"⚠️ 청크 {chunk_idx + 1} 임베딩 생성 실패")
                    continue
                
                # 메타데이터 생성
                metadata = {
                    "source": source_uri,
                    "parentText": original_text,
                    "relatedContent": []
                }
                
                # 이미지 참조 추가
                if image_s3_uri:
                    metadata["relatedContent"].append({
                        "locationType": "S3",
                        "s3Location": {
                            "uri": image_s3_uri
                        }
                    })
                
                # 문서 생성
                doc = {
                    "id": str(uuid.uuid4()),
                    "AMAZON_BEDROCK_TEXT": chunk,
                    "AMAZON_BEDROCK_METADATA": json.dumps(metadata, ensure_ascii=False),
                    "bedrock-knowledge-base-default-vector": embedding,
                    "x-amz-bedrock-kb-source-uri": source_uri,
                    "x-amz-bedrock-kb-document-page-number": page_number,
                    "x-amz-bedrock-kb-data-source-id": "MANUAL_UPLOAD",
                    "title_extracted": self.extract_title(chunk),
                    "category": "자동차매뉴얼",
                    "content_length": len(chunk),
                    "timestamp": datetime.now().isoformat()
                }
                
                # 이미지 URI를 별도 필드로도 저장 (검색 편의성)
                if image_s3_uri:
                    doc["page_image_uri"] = image_s3_uri
                
                # OpenSearch에 인덱싱
                response = self.oss_client.index(
                    index=self.config['index_name'],
                    body=doc
                )
                
                print(f"  ✅ 청크 {chunk_idx + 1}/{len(chunks)} 인덱싱 완료")
                
                # API 호출 제한을 위한 짧은 대기
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ 청크 {chunk_idx + 1} 인덱싱 실패: {e}")
    
    def extract_title(self, text: str) -> str:
        """텍스트에서 제목 추출"""
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
            elif line.startswith('## '):
                return line[3:].strip()
        
        # 첫 번째 문장을 제목으로 사용
        first_sentence = text.split('.')[0].strip()
        return first_sentence[:100] if len(first_sentence) > 100 else first_sentence
    
    def process_document(self, pdf_path: str):
        """
        전체 문서 처리 파이프라인
        
        Args:
            pdf_path (str): PDF 파일 경로
        """
        print(f"🚀 문서 처리 시작: {pdf_path}")
        
        # 0. S3 버킷 생성 및 파일 업로드
        if not self.create_s3_bucket():
            print("❌ S3 버킷 생성 실패")
            return
        
        s3_uri = self.upload_to_s3(pdf_path)
        if not s3_uri:
            print("❌ S3 업로드 실패")
            return
        
        print(f"📍 기본 소스 URI: {s3_uri}")
        
        # 1. PDF를 이미지로 변환
        pages = self.pdf_to_images(pdf_path)
        if not pages:
            print("❌ PDF 변환 실패")
            return
        
        # 2. 각 페이지 처리 (API 호출 제한 테스트를 위해 10페이지 처리)
        total_chunks = 0
        pdf_filename = Path(pdf_path).name
        max_pages = max(10, len(pages))  # 최대 10페이지 또는 전체 페이지 수 중 작은 값
        
        print(f"📋 총 {len(pages)}페이지 중 {max_pages}페이지 처리 예정")
        print(f"⚠️  Claude API 제한: 1분당 {self.claude_call_limit}회 호출")
        print(f"💡 6번째 호출부터 1분 대기가 발생할 수 있습니다")
        
        for page_info in pages[:max_pages]:
            page_number = page_info['page_number']
            image_base64 = page_info['image_base64']
            
            print(f"\n📄 페이지 {page_number + 1}/{len(pages)} 처리 중...")
            
            # 페이지 이미지를 S3에 저장
            image_s3_uri = self.save_page_image_to_s3(image_base64, page_number, pdf_filename)
            
            # Claude로 파싱
            parsed_text = self.parse_page_with_claude(image_base64, page_number)
            if not parsed_text:
                print(f"⚠️ 페이지 {page_number + 1} 파싱 결과가 비어있음")
                continue
            
            # 텍스트 청킹
            chunks = self.chunk_text(parsed_text)
            if not chunks:
                print(f"⚠️ 페이지 {page_number + 1} 청킹 결과가 비어있음")
                continue
            
            # 페이지별 고유 URI 생성 (S3 URI + 페이지 번호)
            page_uri = f"{s3_uri}#page={page_number + 1}"
            print(f"📍 페이지 {page_number + 1} URI: {page_uri}")
            
            # OpenSearch에 인덱싱 (페이지별 고유 URI 및 이미지 참조 포함)
            self.index_document(chunks, page_uri, page_number, parsed_text, image_s3_uri)
            
            total_chunks += len(chunks)
            
            print(f"✅ 페이지 {page_number + 1} 처리 완료 ({len(chunks)}개 청크)")
        
        print(f"\n🎉 문서 처리 완료!")
        print(f"📊 총 처리 결과:")
        print(f"   - 페이지 수: {len(pages)} (처리: {max_pages}페이지)")
        print(f"   - 총 청크 수: {total_chunks}")
        print(f"   - Claude API 호출: {self.claude_call_count}회")
        print(f"   - 기본 S3 URI: {s3_uri}")
        print(f"   - 페이지별 URI 형식: {s3_uri}#page=N")
        print(f"   - 인덱스: {self.config['index_name']}")
        
        # 설정 파일에 S3 정보 추가
        self.config['s3_bucket'] = self.bucket_name
        self.config['s3_uri'] = s3_uri
        with open('opensearch_config.json', 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(f"💾 S3 정보가 설정 파일에 저장되었습니다")

def main():
    """메인 함수"""
    print("🚀 Step 3: 문서 처리 및 임베딩 생성")
    print("🤖 LLM: Claude 3.7 Sonnet")
    print("🧠 임베딩: Titan2")
    print("📄 문서: data/santafe.pdf")
    
    # 문서 경로 확인
    pdf_path = "data/santafe.pdf"
    if not Path(pdf_path).exists():
        print(f"❌ 문서를 찾을 수 없습니다: {pdf_path}")
        print("💡 data 폴더에 crob_santafe.pdf 파일이 있는지 확인하세요")
        return
    
    try:
        # 문서 처리기 초기화
        processor = DocumentProcessor()
        
        # 문서 처리 실행
        processor.process_document(pdf_path)
        
        print("\n📝 다음 단계:")
        print("1. step4_search_test.py를 실행하여 검색 기능을 테스트하세요")
        print("2. RAG 시스템이 준비되었습니다!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("🔧 해결 방법:")
        print("1. AWS 자격 증명이 올바르게 설정되어 있는지 확인하세요")
        print("2. Bedrock 모델 권한이 있는지 확인하세요")
        print("3. OpenSearch Serverless 권한이 있는지 확인하세요")
        print("4. 필요한 라이브러리가 설치되어 있는지 확인하세요:")
        print("   pip install PyMuPDF opensearch-py")

if __name__ == "__main__":
    main()
