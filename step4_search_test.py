#!/usr/bin/env python3
"""
Step 4: RAG 검색 기능 테스트
OpenSearch Serverless에서 벡터 검색을 수행하고 결과를 확인합니다.
"""

import json
import boto3
from pathlib import Path
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import numpy as np

class RAGSearcher:
    def __init__(self):
        """RAG 검색기 초기화"""
        self.config = self.load_config()
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.config['region'])
        self.opensearch_client = self.setup_opensearch_client()
        
    def load_config(self):
        """설정 파일 로드"""
        config_path = Path("opensearch_config.json")
        if not config_path.exists():
            raise FileNotFoundError("opensearch_config.json 파일을 찾을 수 없습니다. step1을 먼저 실행하세요.")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def setup_opensearch_client(self):
        """OpenSearch 클라이언트 설정"""
        session = boto3.Session()
        credentials = session.get_credentials()
        
        auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            self.config['region'],
            'aoss',
            session_token=credentials.token
        )
        
        client = OpenSearch(
            hosts=[{
                'host': self.config['endpoint'].replace('https://', ''),
                'port': 443
            }],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            pool_maxsize=20
        )
        
        return client
    
    def get_embedding(self, text):
        """텍스트를 임베딩으로 변환"""
        try:
            response = self.bedrock_client.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=json.dumps({
                    "inputText": text,
                    "dimensions": 1024,
                    "normalize": True
                })
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['embedding']
            
        except Exception as e:
            print(f"❌ 임베딩 생성 오류: {e}")
            return None
    
    def search_documents(self, query, k=5):
        """문서 검색 수행"""
        print(f"🔍 검색 쿼리: '{query}'")
        
        # 쿼리 임베딩 생성
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        # 벡터 검색 쿼리 구성
        search_body = {
            "size": k,
            "query": {
                "knn": {
                    "bedrock-knowledge-base-default-vector": {
                        "vector": query_embedding,
                        "k": k
                    }
                }
            },
            "_source": {
                "excludes": ["bedrock-knowledge-base-default-vector"]
            }
        }
        
        try:
            # 검색 실행
            response = self.opensearch_client.search(
                index=self.config['index_name'],
                body=search_body
            )
            
            return self.format_search_results(response)
            
        except Exception as e:
            print(f"❌ 검색 오류: {e}")
            return []
    
    def format_search_results(self, response):
        """검색 결과 포맷팅"""
        results = []
        hits = response.get('hits', {}).get('hits', [])
        
        for i, hit in enumerate(hits, 1):
            source = hit['_source']
            score = hit['_score']
            
            result = {
                'rank': i,
                'score': score,
                'content': source.get('AMAZON_BEDROCK_TEXT', ''),
                'metadata': source.get('AMAZON_BEDROCK_METADATA', {}),
                'source_uri': source.get('x-amz-bedrock-kb-source-uri', ''),
                'page_number': source.get('x-amz-bedrock-kb-document-page-number', 'N/A'),
                'page_image_uri': source.get('page_image_uri', '')
            }
            results.append(result)
        
        return results
    
    def display_results(self, results):
        """검색 결과 출력"""
        if not results:
            print("❌ 검색 결과가 없습니다.")
            return
        
        print(f"\n📊 검색 결과 ({len(results)}개):")
        print("=" * 80)
        
        for result in results:
            print(f"\n🏆 순위: {result['rank']}")
            print(f"📊 점수: {result['score']:.4f}")
            print(f"📄 페이지: {result['page_number'] + 1 if isinstance(result['page_number'], int) else result['page_number']}")
            print(f"🔗 소스: {result['source_uri']}")
            
            # 이미지 정보 표시
            if result.get('page_image_uri'):
                print(f"🖼️ 페이지 이미지: {result['page_image_uri']}")
            
            print(f"📝 내용:")
            print("-" * 40)
            # 내용을 적절한 길이로 자르기
            content = result['content']
            if len(content) > 300:
                content = content[:300] + "..."
            print(content)
            print("-" * 40)
    
    def interactive_search(self):
        """대화형 검색 모드"""
        print("🎯 대화형 검색 모드 시작")
        print("💡 검색하고 싶은 내용을 입력하세요 (종료: 'quit' 또는 'exit')")
        print("=" * 60)
        
        while True:
            try:
                query = input("\n🔍 검색어 입력: ").strip()
                
                if query.lower() in ['quit', 'exit', '종료']:
                    print("👋 검색을 종료합니다.")
                    break
                
                if not query:
                    print("❌ 검색어를 입력해주세요.")
                    continue
                
                # 검색 수행
                results = self.search_documents(query)
                self.display_results(results)
                
            except KeyboardInterrupt:
                print("\n👋 검색을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")

def main():
    """메인 함수"""
    print("🚀 Step 4: RAG 검색 기능 테스트")
    print("🔍 벡터 검색 및 결과 확인")
    
    try:
        # 검색기 초기화
        searcher = RAGSearcher()
        
        print(f"🌍 리전: {searcher.config['region']}")
        print(f"🔗 OpenSearch 엔드포인트: {searcher.config['endpoint']}")
        print(f"📄 인덱스: {searcher.config['index_name']}")
        
        # 샘플 검색 테스트
        print("\n🧪 샘플 검색 테스트:")
        sample_queries = [
            "차량 안전 기능",
            "엔진 오일 교환",
            "타이어 점검 방법",
            "브레이크 시스템"
        ]
        
        for query in sample_queries:
            print(f"\n{'='*60}")
            results = searcher.search_documents(query, k=3)
            searcher.display_results(results)
        
        # 대화형 검색 모드
        print(f"\n{'='*60}")
        searcher.interactive_search()
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("🔧 해결 방법:")
        print("1. step1, step2, step3이 모두 성공적으로 실행되었는지 확인하세요")
        print("2. AWS 자격 증명이 올바르게 설정되어 있는지 확인하세요")
        print("3. OpenSearch Serverless 권한이 있는지 확인하세요")

if __name__ == "__main__":
    main()
