"""
2단계: OpenSearch 인덱스 생성
이 스크립트는 OpenSearch Serverless 컬렉션에 벡터 검색용 인덱스를 생성합니다.
optimized-index-with-nori.json 파일을 사용하여 한국어 최적화된 인덱스를 생성합니다.
Titan2 임베딩 모델 (1024차원)에 최적화되었습니다.
"""

import json
import boto3
import time
from opensearchpy import (
    OpenSearch,
    RequestsHttpConnection,
    AWSV4SignerAuth,
    RequestError,
)

class OpenSearchIndexManager:
    def __init__(self, endpoint, region_name='us-west-2'):
        """
        OpenSearch 인덱스 관리자 초기화
        knowledge_base.py의 OpenSearch 클라이언트 초기화 방식을 정확히 참고
        
        Args:
            endpoint (str): OpenSearch Serverless 엔드포인트 URL
            region_name (str): AWS 리전 이름 (기본값: us-west-2)
        """
        self.endpoint = endpoint
        self.region_name = region_name
        
        # AWS 인증 설정 (knowledge_base.py와 동일한 방식)
        credentials = boto3.Session().get_credentials()
        self.awsauth = AWSV4SignerAuth(credentials, region_name, "aoss")
        
        # OpenSearch 클라이언트 초기화 (knowledge_base.py와 정확히 동일한 설정)
        self.oss_client = OpenSearch(
            hosts=[{"host": endpoint.replace('https://', ''), "port": 443}],
            http_auth=self.awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300,  # knowledge_base.py와 동일한 타임아웃
        )
        
        print(f"🔗 OpenSearch 클라이언트 초기화 완료")
        print(f"🌍 리전: {region_name}")
        print(f"🔗 엔드포인트: {endpoint}")
    
    def load_index_template(self, template_file='./optimized-index-with-nori.json'):
        """
        인덱스 템플릿 파일 로드
        
        Args:
            template_file (str): 인덱스 템플릿 JSON 파일 경로
            
        Returns:
            dict: 인덱스 설정 및 매핑
        """
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                template = json.load(f)
            
            print(f"📄 인덱스 템플릿 로드 완료: {template_file}")
            
            # 템플릿에서 실제 인덱스 설정 추출
            # 파일 구조: {"bedrock-knowledge-base-optimized-index": {...}}
            template_key = list(template.keys())[0]
            index_config = template[template_key]
            
            print(f"📋 템플릿 키: {template_key}")
            print(f"🔧 설정 확인:")
            print(f"   - Nori 분석기: {'korean_analyzer' in str(index_config)}")
            print(f"   - KNN 활성화: {index_config.get('settings', {}).get('index', {}).get('knn', 'false')}")
            
            # 벡터 필드 확인
            vector_field = None
            properties = index_config.get('mappings', {}).get('properties', {})
            for field_name, field_config in properties.items():
                if field_config.get('type') == 'knn_vector':
                    vector_field = field_name
                    dimension = field_config.get('dimension', 'unknown')
                    print(f"   - 벡터 필드: {field_name} ({dimension}차원)")
                    break
            
            return index_config
            
        except FileNotFoundError:
            print(f"❌ 템플릿 파일을 찾을 수 없습니다: {template_file}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 오류: {e}")
            raise
        except Exception as e:
            print(f"❌ 템플릿 로드 실패: {e}")
            raise
    
    def create_vector_index_from_template(self, index_name, template_file='./optimized-index-with-nori.json'):
        """
        템플릿 파일을 사용하여 OpenSearch Serverless 벡터 인덱스 생성
        
        Args:
            index_name (str): 인덱스 이름
            template_file (str): 인덱스 템플릿 JSON 파일 경로
        """
        print(f"📦 벡터 인덱스 '{index_name}' 생성 중...")
        print(f"📄 템플릿 파일: {template_file}")
        
        # 템플릿 로드
        index_config = self.load_index_template(template_file)
        
        # 인덱스 생성 (knowledge_base.py와 정확히 동일한 방식)
        try:
            response = self.oss_client.indices.create(
                index=index_name, 
                body=json.dumps(index_config)
            )
            print("✅ 인덱스 생성 완료:")
            print(f"   - 인덱스 이름: {index_name}")
            print(f"   - 템플릿: {template_file}")
            print(f"   - 응답: {response}")
            
            # knowledge_base.py처럼 인덱스 생성 후 대기
            print("⏳ 인덱스 초기화 대기 중...")
            self.interactive_sleep(60)  # knowledge_base.py와 동일한 대기 시간
            
            return True
            
        except RequestError as e:
            # 더 자세한 에러 정보 출력
            print(f"❌ 인덱스 생성 중 오류 발생:")
            print(f"   - 오류 타입: {e.error}")
            print(f"   - 상세 정보: {e.info}")
            print("💡 해결 방법:")
            
            # OpenSearch Serverless 특정 오류 처리
            if 'mapper_parsing_exception' in str(e.error):
                print("   - OpenSearch Serverless에서 지원하지 않는 매핑이 있을 수 있습니다")
                print("   - 템플릿 파일의 설정을 확인하세요")
                print("   - space_type, 분석기 설정 등을 점검하세요")
            elif 'analysis' in str(e.error):
                print("   - 분석기 설정에 문제가 있을 수 있습니다")
                print("   - OpenSearch Serverless에서 지원하는 분석기를 사용하세요")
            elif 'resource_already_exists_exception' in str(e.error):
                print("   - 인덱스가 이미 존재합니다")
                print(f"   - 기존 인덱스 삭제: oss_client.indices.delete(index='{index_name}')")
            else:
                print("   - 템플릿 파일의 설정을 확인하세요")
                print("   - OpenSearch Serverless 호환성을 점검하세요")
            
            return False
        except Exception as e:
            print(f"❌ 예상치 못한 오류 발생: {e}")
            return False
    
    def interactive_sleep(self, seconds):
        """
        knowledge_base.py의 interactive_sleep 함수와 정확히 동일
        시각적 피드백과 함께 대기
        
        Args:
            seconds (int): 대기할 초 수
        """
        dots = ""
        for i in range(seconds):
            dots += "."
            print(f"대기 중{dots}", end="\r")
            time.sleep(1)
        print()  # 새 줄로 이동
    
    def verify_index(self, index_name):
        """
        인덱스 생성 및 설정 확인
        
        Args:
            index_name (str): 인덱스 이름
            
        Returns:
            dict: 인덱스 정보
        """
        try:
            print(f"🔍 인덱스 '{index_name}' 정보 확인 중...")
            
            # 인덱스 존재 확인
            if not self.oss_client.indices.exists(index=index_name):
                print(f"❌ 인덱스 '{index_name}'이 존재하지 않습니다")
                return {'name': index_name, 'exists': False}
            
            # 인덱스 매핑 및 설정 확인
            mapping = self.oss_client.indices.get_mapping(index=index_name)
            settings = self.oss_client.indices.get_settings(index=index_name)
            
            index_info = {
                'name': index_name,
                'exists': True,
                'mapping': mapping[index_name]['mappings'],
                'settings': settings[index_name]['settings']
            }
            
            # 벡터 필드 확인
            properties = mapping[index_name]['mappings']['properties']
            vector_fields = []
            for field_name, field_config in properties.items():
                if field_config.get('type') == 'knn_vector':
                    dimension = field_config.get('dimension')
                    vector_fields.append({'name': field_name, 'dimension': dimension})
                    print(f"✅ 벡터 필드 확인: {field_name} ({dimension}차원)")
            
            if vector_fields:
                # Titan2 호환성 확인
                titan2_compatible = any(vf['dimension'] == 1024 for vf in vector_fields)
                if titan2_compatible:
                    print("✅ Titan2 임베딩 모델과 호환됩니다")
                else:
                    print("⚠️ Titan2 임베딩 모델(1024차원)과 호환되지 않을 수 있습니다")
            else:
                print("⚠️ 벡터 필드가 올바르게 설정되지 않았습니다")
            
            # Nori 분석기 확인
            analyzers = settings[index_name]['settings'].get('index', {}).get('analysis', {}).get('analyzer', {})
            if 'korean_analyzer' in analyzers:
                analyzer_type = analyzers['korean_analyzer'].get('type', 'unknown')
                print(f"✅ 한국어 분석기 확인: korean_analyzer (type: {analyzer_type})")
            else:
                print("⚠️ 한국어 분석기가 설정되지 않았습니다")
            
            # 텍스트 필드 확인
            text_fields = ['AMAZON_BEDROCK_TEXT', 'AMAZON_BEDROCK_TEXT_CHUNK', 'title_extracted']
            for field_name in text_fields:
                if field_name in properties and properties[field_name].get('type') == 'text':
                    analyzer = properties[field_name].get('analyzer', 'default')
                    print(f"✅ 텍스트 필드 확인: {field_name} (analyzer: {analyzer})")
            
            return index_info
            
        except Exception as e:
            print(f"❌ 인덱스 확인 중 오류: {e}")
            return {'name': index_name, 'exists': False, 'error': str(e)}

def load_config():
    """
    설정 파일 로드
    
    Returns:
        dict: 설정 정보
    """
    try:
        with open('opensearch_config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ opensearch_config.json 파일을 찾을 수 없습니다")
        print("먼저 step1_create_opensearch_collection.py를 실행하세요")
        return None
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        return None

def main():
    """
    메인 함수 - OpenSearch 인덱스 생성 실행
    """
    print("🚀 OpenSearch 벡터 인덱스 생성 시작...")
    print("🤖 Titan2 임베딩 모델용 인덱스 생성")
    print("🇰🇷 Nori 한국어 분석기 사용")
    print("📄 사용자 정의 템플릿만 사용")
    
    # 설정 로드
    config = load_config()
    if not config:
        return
    
    # 설정 정보 출력
    print(f"📋 컬렉션: {config['collection_name']}")
    print(f"🔗 엔드포인트: {config['endpoint']}")
    print(f"🌍 리전: {config['region']}")
    
    try:
        # OpenSearch 인덱스 관리자 생성
        manager = OpenSearchIndexManager(
            endpoint=config['endpoint'],
            region_name=config['region']
        )
        
        # 인덱스 설정
        INDEX_NAME = f"{config['collection_name']}-index"  # 컬렉션명-index
        TEMPLATE_FILE = './optimized-index-with-nori.json'
        
        print(f"📝 생성할 인덱스: {INDEX_NAME}")
        print(f"📄 템플릿 파일: {TEMPLATE_FILE}")
        
        # 인덱스 생성 (사용자 정의 템플릿만 사용)
        success = manager.create_vector_index_from_template(
            index_name=INDEX_NAME,
            template_file=TEMPLATE_FILE
        )
        
        if success:
            # 인덱스 확인
            index_info = manager.verify_index(INDEX_NAME)
            
            # 결과 출력
            print("\n" + "="*60)
            print("🎉 OpenSearch 벡터 인덱스 생성 완료!")
            print("="*60)
            print(f"인덱스 이름: {INDEX_NAME}")
            print(f"템플릿 파일: {TEMPLATE_FILE}")
            print(f"엔드포인트: {config['endpoint']}")
            print(f"리전: {config['region']}")
            print(f"상태: {'생성됨' if index_info['exists'] else '생성 실패'}")
            print("특징:")
            print("  - 🇰🇷 Nori 한국어 분석기 적용")
            print("  - 🤖 Titan2 임베딩 모델 호환 (1024차원)")
            print("  - 🔍 하이브리드 검색 지원 (벡터 + 키워드)")
            print("  - 📄 사용자 정의 템플릿 사용")
            print("="*60)
            
            # 설정 파일 업데이트
            config['index_name'] = INDEX_NAME
            config['vector_dimension'] = 1024
            config['embedding_model'] = 'amazon.titan-embed-text-v2:0'
            config['template_file'] = TEMPLATE_FILE
            config['features'] = {
                'nori_analyzer': True,
                'korean_optimized': True,
                'hybrid_search': True,
                'titan2_compatible': True,
                'user_defined_template': True
            }
            
            with open('opensearch_config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print("💾 설정 정보가 업데이트되었습니다")
            
            # 다음 단계 안내
            print("\n📝 다음 단계:")
            print("1. step3_document_processing.py를 실행하여 문서를 처리하세요")
            print("2. Titan2 임베딩 모델로 문서 임베딩을 생성하세요")
            print(f"3. 인덱스 이름: {INDEX_NAME}")
            print("4. 한국어 최적화된 벡터 검색이 준비되었습니다!")
            print("5. 임베딩 모델: amazon.titan-embed-text-v2:0 (1024차원)")
            print("6. 분석기: Nori 한국어 분석기")
            
        else:
            print("❌ 인덱스 생성에 실패했습니다")
            print("💡 해결 방법:")
            print("1. optimized-index-with-nori.json 파일이 존재하는지 확인하세요")
            print("2. 템플릿 파일의 JSON 형식이 올바른지 확인하세요")
            print("3. OpenSearch Serverless 호환성을 확인하세요")
            print("4. 오류 메시지를 참고하여 템플릿을 수정하세요")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("🔧 해결 방법:")
        print("1. AWS 자격 증명이 올바르게 설정되어 있는지 확인하세요")
        print("2. OpenSearch Serverless 권한이 있는지 확인하세요")
        print("3. 엔드포인트 URL이 올바른지 확인하세요")
        print("4. us-west-2 리전에 대한 권한이 있는지 확인하세요")
        print("5. optimized-index-with-nori.json 파일이 존재하는지 확인하세요")
        print("6. 네트워크 연결 상태를 확인하세요")
        print("7. 필요한 라이브러리가 설치되어 있는지 확인하세요:")
        print("   pip install opensearch-py")

if __name__ == "__main__":
    main()
