#!/usr/bin/env python3
"""
Streamlit RAG 채팅 앱
하이브리드 검색과 Claude 3.5 Sonnet 스트리밍 응답을 제공하는 웹 인터페이스
"""

import streamlit as st
import json
import boto3
from pathlib import Path
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from typing import List, Dict, Any, Generator
import time
import re

# 페이지 설정
st.set_page_config(
    page_title="🚗 자동차 매뉴얼 AI 어시스턴트",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitRAGChatbot:
    def __init__(self, config_file='opensearch_config.json'):
        """Streamlit RAG 채팅봇 초기화"""
        self.config = self.load_config(config_file)
        if not self.config:
            st.error("설정 파일을 로드할 수 없습니다")
            st.stop()
        
        # AWS 클라이언트 초기화
        self.region = self.config['region']
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.region)
        self.s3_client = boto3.client('s3', region_name=self.region)  # S3 클라이언트 추가
        self.opensearch_client = self.setup_opensearch_client()
        
        # Claude 3.5 Sonnet 모델 ID
        self.claude_model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        
        # 시스템 프롬프트
        self.system_prompt = """당신은 자동차 매뉴얼 전문 AI 어시스턴트입니다. 
사용자의 질문에 대해 하이브리드 검색으로 찾은 문서 내용을 바탕으로 정확하고 도움이 되는 답변을 제공하세요.

답변 가이드라인:
1. 제공된 문서 내용만을 바탕으로 답변하세요
2. 문서에 없는 내용은 추측하지 마세요
3. 구체적인 절차나 수치가 있다면 정확히 인용하세요
4. 답변 마지막에 참조한 페이지 정보를 포함하세요
5. 한국어로 친근하고 이해하기 쉽게 답변하세요

답변 형식:
- 질문에 대한 직접적인 답변
- 구체적인 설명이나 절차 (단계별로 설명)
- 주의사항이나 추가 정보
- 참조: 페이지 X"""
    
    @st.cache_data
    def load_config(_self, config_file: str) -> Dict:
        """설정 파일 로드 (캐시됨)"""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                return None
            
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"설정 파일 로드 오류: {e}")
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
    
    def generate_presigned_url(self, s3_uri: str, expiration: int = 3600) -> str:
        """S3 URI를 presigned URL로 변환"""
        try:
            # S3 URI 파싱 (s3://bucket/key 형태)
            if not s3_uri.startswith('s3://'):
                return None
            
            # s3:// 제거하고 bucket과 key 분리
            s3_path = s3_uri[5:]  # s3:// 제거
            bucket_name, key = s3_path.split('/', 1)
            
            # presigned URL 생성
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': key},
                ExpiresIn=expiration
            )
            
            return presigned_url
            
        except Exception as e:
            st.error(f"Presigned URL 생성 오류: {e}")
            return None
    
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
            st.error(f"임베딩 생성 오류: {e}")
            return None
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """벡터 + 렉시컬 검색을 별도로 수행하여 결합"""
        all_results = []
        
        # 1. 벡터 검색 수행
        vector_results = self.vector_search(query, k)
        
        # 2. 렉시컬 검색 수행
        lexical_results = self.lexical_search_only(query, k)
        
        # 3. 결과 결합 및 중복 제거
        seen_ids = set()
        
        # 벡터 검색 결과 추가 (높은 가중치)
        for result in vector_results:
            result_id = f"{result['source_uri']}_{result['page_number']}"
            if result_id not in seen_ids:
                result['search_type'] = 'vector'
                result['combined_score'] = result['score'] * 1.2  # 벡터 검색에 가중치
                all_results.append(result)
                seen_ids.add(result_id)
        
        # 렉시컬 검색 결과 추가
        for result in lexical_results:
            result_id = f"{result['source_uri']}_{result['page_number']}"
            if result_id not in seen_ids:
                result['search_type'] = 'lexical'
                result['combined_score'] = result['score']
                all_results.append(result)
                seen_ids.add(result_id)
            else:
                # 이미 있는 결과라면 점수 조합
                for existing in all_results:
                    existing_id = f"{existing['source_uri']}_{existing['page_number']}"
                    if existing_id == result_id:
                        existing['combined_score'] = (existing['combined_score'] + result['score']) / 2
                        existing['search_type'] = 'hybrid'
                        break
        
        # 결합된 점수로 정렬
        all_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return all_results[:k]
    
    def vector_search(self, query: str, k: int = 5) -> List[Dict]:
        """벡터 검색만 수행"""
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
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
            response = self.opensearch_client.search(
                index=self.config['index_name'],
                body=search_body
            )
            
            return self.format_search_results(response)
            
        except Exception as e:
            st.error(f"벡터 검색 오류: {e}")
            return []
    
    def lexical_search_only(self, query: str, k: int = 5) -> List[Dict]:
        """렉시컬 검색만 수행"""
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
            response = self.opensearch_client.search(
                index=self.config['index_name'],
                body=search_body
            )
            
            return self.format_search_results(response)
            
        except Exception as e:
            st.error(f"렉시컬 검색 오류: {e}")
            return []
    
    def format_search_results(self, response: Dict) -> List[Dict]:
        """검색 결과 포맷팅"""
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
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    def stream_claude_response(self, user_question: str, context: str) -> Generator[str, None, None]:
        """Claude 3.5 Sonnet 스트리밍 응답 생성"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": f"""사용자 질문: {user_question}

{context}

위 문서 내용을 바탕으로 사용자의 질문에 답변해주세요."""
                }
            ]
            
            # 스트리밍 요청
            response = self.bedrock_client.invoke_model_with_response_stream(
                modelId=self.claude_model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2000,
                    "system": self.system_prompt,
                    "messages": messages,
                    "temperature": 0.1
                })
            )
            
            # 스트리밍 응답 처리
            for event in response['body']:
                chunk = json.loads(event['chunk']['bytes'])
                
                if chunk['type'] == 'content_block_delta':
                    if 'delta' in chunk and 'text' in chunk['delta']:
                        yield chunk['delta']['text']
                        
        except Exception as e:
            yield f"❌ 답변 생성 오류: {e}"

# Streamlit 앱 초기화
@st.cache_resource
def init_chatbot():
    """채팅봇 초기화 (캐시됨)"""
    return StreamlitRAGChatbot()

def display_search_results(search_results: List[Dict], chatbot):
    """검색 결과 표시 (이미지 포함)"""
    if not search_results:
        st.warning("검색 결과가 없습니다.")
        return
    
    st.subheader(f"🔍 검색 결과 ({len(search_results)}개)")
    
    for i, result in enumerate(search_results):
        # 검색 타입 표시
        search_type_emoji = {
            'vector': '🧠',
            'lexical': '🔍', 
            'hybrid': '⚡'
        }
        search_type = result.get('search_type', 'unknown')
        type_emoji = search_type_emoji.get(search_type, '❓')
        
        with st.expander(f"{type_emoji} {result.get('title', '제목 없음')} - 페이지 {result['page_number'] + 1 if isinstance(result['page_number'], int) else result['page_number']} (점수: {result.get('combined_score', result['score']):.3f})"):
            
            # 기본 정보
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 검색 타입 배지
                if search_type != 'unknown':
                    st.badge(f"{search_type.upper()} 검색")
                
                if result.get('category'):
                    st.badge(result['category'])
                
                # 하이라이트된 텍스트 표시
                if result.get('highlighted_snippets'):
                    st.markdown("**🔍 매칭 구간:**")
                    for snippet in result['highlighted_snippets'][:2]:
                        # HTML 태그를 마크다운으로 변환
                        clean_snippet = snippet.replace('<em>', '**').replace('</em>', '**')
                        st.markdown(f"• {clean_snippet}")
                else:
                    # 일반 미리보기
                    preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                    st.markdown(f"**📝 내용 미리보기:**\n{preview}")
            
            with col2:
                # 이미지 표시
                if result.get('page_image_uri'):
                    st.markdown("**🖼️ 페이지 이미지:**")
                    try:
                        # S3 presigned URL 생성
                        presigned_url = chatbot.generate_presigned_url(result['page_image_uri'])
                        
                        if presigned_url:
                            # 이미지 직접 표시
                            st.image(presigned_url, caption=f"페이지 {result['page_number'] + 1 if isinstance(result['page_number'], int) else result['page_number']}", width=200)
                        else:
                            st.caption("이미지를 불러올 수 없습니다")
                            st.code(result['page_image_uri'])
                    except Exception as e:
                        st.caption(f"이미지 로딩 오류: {e}")
                        st.code(result['page_image_uri'])

def main():
    """메인 앱"""
    # 헤더
    st.title("🚗 자동차 매뉴얼 AI 어시스턴트")
    st.markdown("**벡터 + 렉시컬 검색 결합 + Claude 3.5 Sonnet 스트리밍 응답**")
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 검색 설정
        max_results = st.slider("검색 결과 수", 1, 10, 5)
        show_search_details = st.checkbox("검색 결과 상세 보기", True)
        
        st.markdown("---")
        st.markdown("### 📊 시스템 정보")
        
        # 채팅봇 초기화
        try:
            chatbot = init_chatbot()
            st.success("✅ 시스템 준비 완료")
            st.info(f"🌍 리전: {chatbot.region}")
            st.info(f"📄 인덱스: {chatbot.config['index_name']}")
        except Exception as e:
            st.error(f"❌ 시스템 초기화 실패: {e}")
            st.stop()
    
    # 채팅 히스토리 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.search_results = []
    
    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 검색 결과가 있는 경우 표시
            if message["role"] == "assistant" and "search_results" in message:
                if show_search_details and message["search_results"]:
                    display_search_results(message["search_results"], chatbot)
    
    # 사용자 입력
    if prompt := st.chat_input("자동차 매뉴얼에 대해 무엇이든 물어보세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            # 검색 수행
            with st.spinner("🔍 관련 문서 검색 중..."):
                search_results = chatbot.search_documents(prompt, max_results)
            
            if not search_results:
                st.error("관련된 문서를 찾을 수 없습니다. 다른 질문을 시도해보세요.")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "관련된 문서를 찾을 수 없습니다. 다른 질문을 시도해보세요.",
                    "search_results": []
                })
            else:
                # 컨텍스트 구성
                context = chatbot.build_context(search_results)
                
                # 스트리밍 응답 표시
                response_placeholder = st.empty()
                full_response = ""
                
                with st.spinner("🤖 Claude 3.5 Sonnet이 답변을 생성하고 있습니다..."):
                    for chunk in chatbot.stream_claude_response(prompt, context):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                
                # 최종 응답 표시
                response_placeholder.markdown(full_response)
                
                # 검색 결과 표시
                if show_search_details:
                    display_search_results(search_results, chatbot)
                
                # 세션에 저장
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "search_results": search_results
                })
    
    # 하단 정보
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💬 대화 수", len([m for m in st.session_state.messages if m["role"] == "user"]))
    
    with col2:
        if st.session_state.messages:
            last_search = next((m["search_results"] for m in reversed(st.session_state.messages) if "search_results" in m), [])
            st.metric("🔍 마지막 검색 결과", len(last_search))
    
    with col3:
        if st.button("🗑️ 대화 초기화"):
            st.session_state.messages = []
            st.session_state.search_results = []
            st.rerun()

if __name__ == "__main__":
    main()
