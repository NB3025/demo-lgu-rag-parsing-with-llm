#!/usr/bin/env python3
"""
Step 5: RAG 채팅 시스템
OpenSearch Serverless에서 검색한 결과를 바탕으로 Claude 3.5 Sonnet이 답변을 생성합니다.
"""

import json
import boto3
from pathlib import Path
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from typing import List, Dict, Any
import time

class RAGChatbot:
    def __init__(self, config_file='opensearch_config.json'):
        """
        RAG 채팅봇 초기화
        
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
        self.opensearch_client = self.setup_opensearch_client()
        
        # Claude 3.5 Sonnet 모델 ID
        self.claude_model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        
        # 시스템 프롬프트 로드
        self.system_prompt = self.load_system_prompt()
        
        print("🤖 RAG 채팅봇 초기화 완료")
        print(f"🌍 리전: {self.region}")
        print(f"🔗 OpenSearch 엔드포인트: {self.config['endpoint']}")
        print(f"📄 인덱스: {self.config['index_name']}")
        print(f"🧠 LLM: Claude 3.5 Sonnet")
        
    def load_config(self, config_file: str) -> Dict:
        """설정 파일 로드"""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                print(f"❌ 설정 파일을 찾을 수 없습니다: {config_file}")
                return None
            
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 설정 파일 로드 오류: {e}")
            return None
    
    def setup_opensearch_client(self):
        """OpenSearch 클라이언트 설정"""
        session = boto3.Session()
        credentials = session.get_credentials()
        
        auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            self.region,
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
    
    def load_system_prompt(self) -> str:
        """시스템 프롬프트 로드"""
        return """당신은 자동차 매뉴얼 전문 AI 어시스턴트입니다. 
사용자의 질문에 대해 하이브리드 검색(벡터 + 키워드)으로 찾은 문서 내용을 바탕으로 정확하고 도움이 되는 답변을 제공하세요.

답변 가이드라인:
1. 제공된 문서 내용만을 바탕으로 답변하세요
2. 문서에 없는 내용은 추측하지 마세요
3. 키워드 매칭과 의미적 유사성을 모두 고려한 검색 결과를 활용하세요
4. 답변 마지막에 참조한 페이지 정보를 포함하세요
5. 관련 이미지가 있다면 언급하세요
6. 한국어로 친근하고 이해하기 쉽게 답변하세요
7. 구체적인 절차나 수치가 있다면 정확히 인용하세요

답변 형식:
- 질문에 대한 직접적인 답변
- 구체적인 설명이나 절차 (단계별로 설명)
- 주의사항이나 추가 정보
- 참조: 페이지 X (이미지 포함)

검색 품질:
- 벡터 검색: 의미적으로 유사한 내용 발견
- 키워드 검색: 정확한 용어 매칭
- 하이브리드: 두 방식의 장점을 결합한 최적의 결과"""
    
    def get_embedding(self, text: str) -> List[float]:
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
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """하이브리드 검색 수행 (벡터 + 렉시컬)"""
        print(f"🔍 하이브리드 검색 중: '{query}'")
        
        # 쿼리 임베딩 생성
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            print("❌ 임베딩 생성 실패, 렉시컬 검색만 수행")
            return self.lexical_search_only(query, k)
        
        # 하이브리드 검색 쿼리 구성
        search_body = {
            "size": k,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            # 벡터 검색 (의미적 유사성)
                            "knn": {
                                "bedrock-knowledge-base-default-vector": {
                                    "vector": query_embedding,
                                    "k": k * 2  # 더 많은 후보 확보
                                }
                            }
                        },
                        {
                            # 렉시컬 검색 (키워드 매칭)
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "AMAZON_BEDROCK_TEXT^3",      # 본문 텍스트 (가중치 3배)
                                    "title_extracted^2",          # 제목 (가중치 2배)
                                    "category"                    # 카테고리
                                ],
                                "type": "best_fields",
                                "fuzziness": "AUTO",              # 오타 허용
                                "analyzer": "nori"               # 한국어 형태소 분석
                            }
                        }
                    ]
                }
            },
            "_source": {
                "excludes": ["bedrock-knowledge-base-default-vector"]
            },
            # 결과 다양성을 위한 추가 설정
            "highlight": {
                "fields": {
                    "AMAZON_BEDROCK_TEXT": {
                        "fragment_size": 150,
                        "number_of_fragments": 2
                    }
                }
            }
        }
        
        try:
            # 하이브리드 검색 실행
            print("🔄 벡터 검색 + 렉시컬 검색 수행 중...")
            response = self.opensearch_client.search(
                index=self.config['index_name'],
                body=search_body
            )
            
            results = self.format_search_results(response)
            print(f"✅ 하이브리드 검색 완료: {len(results)}개 결과")
            return results
            
        except Exception as e:
            print(f"❌ 하이브리드 검색 오류: {e}")
            print("🔄 렉시컬 검색으로 대체 시도...")
            return self.lexical_search_only(query, k)
    
    def lexical_search_only(self, query: str, k: int = 5) -> List[Dict]:
        """렉시컬 검색만 수행 (하이브리드 검색 실패시 대체)"""
        search_body = {
            "size": k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "AMAZON_BEDROCK_TEXT^3",
                        "title_extracted^2",
                        "category"
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                    "analyzer": "nori"
                }
            },
            "_source": {
                "excludes": ["bedrock-knowledge-base-default-vector"]
            },
            "highlight": {
                "fields": {
                    "AMAZON_BEDROCK_TEXT": {
                        "fragment_size": 150,
                        "number_of_fragments": 2
                    }
                }
            }
        }
        
        try:
            print("🔍 렉시컬 검색 수행 중...")
            response = self.opensearch_client.search(
                index=self.config['index_name'],
                body=search_body
            )
            
            results = self.format_search_results(response)
            print(f"✅ 렉시컬 검색 완료: {len(results)}개 결과")
            return results
            
        except Exception as e:
            print(f"❌ 렉시컬 검색 오류: {e}")
            return []
    
    def format_search_results(self, response: Dict) -> List[Dict]:
        """검색 결과 포맷팅 (하이라이트 정보 포함)"""
        results = []
        hits = response.get('hits', {}).get('hits', [])
        
        for i, hit in enumerate(hits, 1):
            source = hit['_source']
            score = hit['_score']
            highlight = hit.get('highlight', {})
            
            result = {
                'rank': i,
                'score': score,
                'content': source.get('AMAZON_BEDROCK_TEXT', ''),
                'source_uri': source.get('x-amz-bedrock-kb-source-uri', ''),
                'page_number': source.get('x-amz-bedrock-kb-document-page-number', 'N/A'),
                'page_image_uri': source.get('page_image_uri', ''),
                'timestamp': source.get('timestamp', ''),
                'title': source.get('title_extracted', ''),
                'category': source.get('category', '')
            }
            
            # 하이라이트 정보 추가
            if highlight:
                highlighted_text = highlight.get('AMAZON_BEDROCK_TEXT', [])
                if highlighted_text:
                    result['highlighted_snippets'] = highlighted_text
            
            results.append(result)
        
        return results
    
    def build_context(self, search_results: List[Dict]) -> str:
        """검색 결과를 바탕으로 컨텍스트 구성"""
        if not search_results:
            return "관련 문서를 찾을 수 없습니다."
        
        context_parts = []
        context_parts.append("=== 관련 문서 내용 ===\n")
        
        for result in search_results:
            page_num = result['page_number']
            page_display = f"페이지 {page_num + 1}" if isinstance(page_num, int) else f"페이지 {page_num}"
            
            context_parts.append(f"[{page_display}]")
            context_parts.append(result['content'])
            
            # 이미지 정보 추가
            if result['page_image_uri']:
                context_parts.append(f"(이미지 참조: {result['page_image_uri']})")
            
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    def generate_answer(self, user_question: str, context: str) -> str:
        """Claude 3.5 Sonnet으로 답변 생성"""
        try:
            # 메시지 구성
            messages = [
                {
                    "role": "user",
                    "content": f"""사용자 질문: {user_question}

{context}

위 문서 내용을 바탕으로 사용자의 질문에 답변해주세요."""
                }
            ]
            
            # Claude 3.5 Sonnet 호출
            response = self.bedrock_client.invoke_model(
                modelId=self.claude_model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2000,
                    "system": self.system_prompt,
                    "messages": messages,
                    "temperature": 0.1
                })
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
            
        except Exception as e:
            print(f"❌ 답변 생성 오류: {e}")
            return "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다."
    
    def format_response(self, answer: str, search_results: List[Dict]) -> Dict:
        """최종 응답 포맷팅"""
        response = {
            'answer': answer,
            'sources': [],
            'search_results_count': len(search_results)
        }
        
        # 소스 정보 추가
        for result in search_results:
            page_num = result['page_number']
            page_display = f"페이지 {page_num + 1}" if isinstance(page_num, int) else f"페이지 {page_num}"
            
            source_info = {
                'page': page_display,
                'source_uri': result['source_uri'],
                'score': result['score'],
                'content_preview': result['content'][:100] + "..." if len(result['content']) > 100 else result['content'],
                'title': result.get('title', ''),
                'category': result.get('category', '')
            }
            
            # 이미지 정보 추가
            if result['page_image_uri']:
                source_info['image_uri'] = result['page_image_uri']
            
            # 하이라이트 정보 추가
            if result.get('highlighted_snippets'):
                source_info['highlights'] = result['highlighted_snippets']
            
            response['sources'].append(source_info)
        
        return response
    
    def chat(self, user_question: str, max_results: int = 5) -> Dict:
        """
        사용자 질문에 대한 RAG 기반 답변 생성
        
        Args:
            user_question (str): 사용자 질문
            max_results (int): 검색할 최대 결과 수
            
        Returns:
            Dict: 답변과 소스 정보
        """
        print(f"\n💬 사용자 질문: {user_question}")
        
        # 1. 문서 검색
        search_results = self.search_documents(user_question, max_results)
        
        if not search_results:
            return {
                'answer': "죄송합니다. 관련된 문서를 찾을 수 없습니다. 다른 질문을 시도해보세요.",
                'sources': [],
                'search_results_count': 0
            }
        
        print(f"📊 검색 결과: {len(search_results)}개 문서 발견")
        
        # 2. 컨텍스트 구성
        context = self.build_context(search_results)
        
        # 3. 답변 생성
        print("🤖 Claude 3.5 Sonnet으로 답변 생성 중...")
        answer = self.generate_answer(user_question, context)
        
        # 4. 응답 포맷팅
        response = self.format_response(answer, search_results)
        
        return response
    
    def interactive_chat(self):
        """대화형 채팅 모드"""
        print("\n🎯 RAG 채팅봇과 대화를 시작합니다!")
        print("💡 자동차 매뉴얼에 대해 무엇이든 물어보세요")
        print("🚪 종료하려면 'quit', 'exit', '종료'를 입력하세요")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n❓ 질문: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                    print("👋 채팅을 종료합니다. 감사합니다!")
                    break
                
                if not user_input:
                    print("❌ 질문을 입력해주세요.")
                    continue
                
                # RAG 답변 생성
                start_time = time.time()
                response = self.chat(user_input)
                end_time = time.time()
                
                # 답변 출력
                print(f"\n🤖 답변:")
                print("=" * 50)
                print(response['answer'])
                print("=" * 50)
                
                # 소스 정보 출력
                if response['sources']:
                    print(f"\n📚 참조 문서 ({len(response['sources'])}개):")
                    for i, source in enumerate(response['sources'], 1):
                        print(f"  {i}. {source['page']} (점수: {source['score']:.3f})")
                        
                        # 제목과 카테고리 표시
                        if source.get('title'):
                            print(f"     📋 제목: {source['title']}")
                        if source.get('category'):
                            print(f"     🏷️ 카테고리: {source['category']}")
                        
                        # 이미지 정보 표시
                        if 'image_uri' in source:
                            print(f"     🖼️ 이미지: {source['image_uri']}")
                        
                        # 하이라이트 정보 표시
                        if source.get('highlights'):
                            print(f"     🔍 매칭 구간:")
                            for highlight in source['highlights'][:2]:  # 최대 2개만 표시
                                # HTML 태그 제거하고 표시
                                clean_highlight = highlight.replace('<em>', '**').replace('</em>', '**')
                                print(f"       • {clean_highlight}")
                        else:
                            print(f"     📝 미리보기: {source['content_preview']}")
                
                print(f"\n⏱️ 응답 시간: {end_time - start_time:.2f}초")
                
            except KeyboardInterrupt:
                print("\n👋 채팅을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")

def main():
    """메인 함수"""
    print("🚀 Step 5: RAG 채팅 시스템 (하이브리드 검색)")
    print("🤖 Claude 3.5 Sonnet 기반 자동차 매뉴얼 어시스턴트")
    print("🔍 벡터 검색 + 렉시컬 검색 결합으로 최적의 검색 성능 제공")
    
    try:
        # RAG 채팅봇 초기화
        chatbot = RAGChatbot()
        
        # 샘플 질문 테스트
        print("\n🧪 샘플 질문 테스트:")
        sample_questions = [
            "차량 안전 기능에 대해 알려주세요",
            "엔진 오일은 언제 교환해야 하나요?",
            "타이어 점검은 어떻게 하나요?"
        ]
        
        for question in sample_questions:
            print(f"\n{'='*60}")
            response = chatbot.chat(question, max_results=3)
            
            print(f"❓ 질문: {question}")
            print(f"🤖 답변: {response['answer'][:200]}...")
            print(f"📚 참조: {len(response['sources'])}개 문서")
        
        # 대화형 모드 시작
        print(f"\n{'='*60}")
        chatbot.interactive_chat()
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("🔧 해결 방법:")
        print("1. step1, step2, step3이 모두 성공적으로 실행되었는지 확인하세요")
        print("2. AWS 자격 증명이 올바르게 설정되어 있는지 확인하세요")
        print("3. Claude 3.5 Sonnet 모델 권한이 있는지 확인하세요")

if __name__ == "__main__":
    main()
