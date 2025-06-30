"""
1단계: OpenSearch Serverless 컬렉션 생성
이 스크립트는 AWS OpenSearch Serverless 컬렉션을 생성합니다.
knowledge_base.py의 OpenSearch 관련 코드를 참고하여 작성되었습니다.
"""

import boto3
import json
import time
import uuid
from botocore.exceptions import ClientError

class OpenSearchServerlessManager:
    def __init__(self, region_name='us-west-2'):
        """
        OpenSearch Serverless 관리자 초기화
        
        Args:
            region_name (str): AWS 리전 이름 (기본값: us-west-2)
        """
        self.region_name = region_name
        self.aoss_client = boto3.client('opensearchserverless', region_name=region_name)
        self.sts_client = boto3.client('sts', region_name=region_name)
        
        # 현재 사용자 정보 가져오기
        self.identity = self.sts_client.get_caller_identity()["Arn"]
        self.account_number = self.sts_client.get_caller_identity().get("Account")
        
        # 고유 suffix 생성 (knowledge_base.py와 동일한 방식)
        self.suffix = str(uuid.uuid4())[:4]
        
        print(f"🔗 OpenSearch Serverless 클라이언트 초기화 완료")
        print(f"🌍 리전: {self.region_name}")
        print(f"👤 현재 사용자 ARN: {self.identity}")
        print(f"🏢 계정 번호: {self.account_number}")
        print(f"🔖 Suffix: {self.suffix}")
    
    def create_policies_in_oss(self, vector_store_name):
        """
        OpenSearch Serverless 암호화, 네트워크 및 데이터 액세스 정책 생성
        항상 새로운 이름으로 생성하여 기존 정책과 충돌하지 않도록 함
        정책 이름은 32자 제한을 준수
        
        Args:
            vector_store_name (str): 벡터 스토어(컬렉션) 이름
            
        Returns:
            tuple: (encryption_policy, network_policy, access_policy)
        """
        # 32자 제한을 고려한 짧은 고유 이름 생성
        short_timestamp = str(int(time.time()))[-6:]  # 마지막 6자리만 사용
        short_uuid = str(uuid.uuid4())[:4]  # UUID 4자리
        
        # 정책 이름을 32자 이내로 제한 (rag-car-manual = 14자)
        encryption_policy_name = f"rcm-enc-{self.suffix}-{short_timestamp}"  # 약 20자
        network_policy_name = f"rcm-net-{self.suffix}-{short_timestamp}"     # 약 20자
        access_policy_name = f"rcm-acc-{self.suffix}-{short_timestamp}"      # 약 20자
        
        print(f"🔐 보안 정책 생성 중...")
        print(f"📝 암호화 정책: {encryption_policy_name} ({len(encryption_policy_name)}자)")
        print(f"📝 네트워크 정책: {network_policy_name} ({len(network_policy_name)}자)")
        print(f"📝 액세스 정책: {access_policy_name} ({len(access_policy_name)}자)")
        
        # 1. 암호화 정책 생성 (항상 새로운 이름으로 생성)
        try:
            encryption_policy = self.aoss_client.create_security_policy(
                name=encryption_policy_name,
                policy=json.dumps({
                    "Rules": [
                        {
                            "Resource": ["collection/" + vector_store_name],
                            "ResourceType": "collection",
                        }
                    ],
                    "AWSOwnedKey": True,
                }),
                type="encryption",
            )
            print("✅ 암호화 정책 생성 완료")
        except self.aoss_client.exceptions.ConflictException:
            # 충돌 시 더 고유한 이름으로 재시도
            encryption_policy_name = f"rcm-enc-{short_uuid}-{short_timestamp}"
            print(f"🔄 정책 이름 충돌, 새로운 이름으로 재시도: {encryption_policy_name}")
            encryption_policy = self.aoss_client.create_security_policy(
                name=encryption_policy_name,
                policy=json.dumps({
                    "Rules": [
                        {
                            "Resource": ["collection/" + vector_store_name],
                            "ResourceType": "collection",
                        }
                    ],
                    "AWSOwnedKey": True,
                }),
                type="encryption",
            )
            print("✅ 암호화 정책 생성 완료 (재시도)")
        
        # 2. 네트워크 정책 생성 (항상 새로운 이름으로 생성)
        try:
            network_policy = self.aoss_client.create_security_policy(
                name=network_policy_name,
                policy=json.dumps([
                    {
                        "Rules": [
                            {
                                "Resource": ["collection/" + vector_store_name],
                                "ResourceType": "collection",
                            },
                            {
                                "Resource": ["collection/" + vector_store_name],
                                "ResourceType": "dashboard"
                            }
                        ],
                        "AllowFromPublic": True,
                    }
                ]),
                type="network",
            )
            print("✅ 네트워크 정책 생성 완료 (OpenSearch 대시보드 액세스 포함)")
        except self.aoss_client.exceptions.ConflictException:
            # 충돌 시 더 고유한 이름으로 재시도
            network_policy_name = f"rcm-net-{short_uuid}-{short_timestamp}"
            print(f"🔄 정책 이름 충돌, 새로운 이름으로 재시도: {network_policy_name}")
            network_policy = self.aoss_client.create_security_policy(
                name=network_policy_name,
                policy=json.dumps([
                    {
                        "Rules": [
                            {
                                "Resource": ["collection/" + vector_store_name],
                                "ResourceType": "collection",
                            },
                            {
                                "Resource": ["collection/" + vector_store_name],
                                "ResourceType": "dashboard"
                            }
                        ],
                        "AllowFromPublic": True,
                    }
                ]),
                type="network",
            )
            print("✅ 네트워크 정책 생성 완료 (재시도)")
        
        # 3. 데이터 액세스 정책 생성 (항상 새로운 이름으로 생성)
        try:
            access_policy = self.aoss_client.create_access_policy(
                name=access_policy_name,
                policy=json.dumps([
                    {
                        "Rules": [
                            {
                                "Resource": ["collection/" + vector_store_name],
                                "Permission": [
                                    "aoss:CreateCollectionItems",
                                    "aoss:DeleteCollectionItems",
                                    "aoss:UpdateCollectionItems",
                                    "aoss:DescribeCollectionItems",
                                ],
                                "ResourceType": "collection",
                            },
                            {
                                "Resource": ["index/" + vector_store_name + "/*"],
                                "Permission": [
                                    "aoss:CreateIndex",
                                    "aoss:DeleteIndex",
                                    "aoss:UpdateIndex",
                                    "aoss:DescribeIndex",
                                    "aoss:ReadDocument",
                                    "aoss:WriteDocument",
                                ],
                                "ResourceType": "index",
                            },
                        ],
                        "Principal": [self.identity],
                        "Description": "Easy data policy",  # knowledge_base.py와 동일한 설명
                    }
                ]),
                type="data",
            )
            print("✅ 데이터 액세스 정책 생성 완료")
        except self.aoss_client.exceptions.ConflictException:
            # 충돌 시 더 고유한 이름으로 재시도
            access_policy_name = f"rcm-acc-{short_uuid}-{short_timestamp}"
            print(f"🔄 정책 이름 충돌, 새로운 이름으로 재시도: {access_policy_name}")
            access_policy = self.aoss_client.create_access_policy(
                name=access_policy_name,
                policy=json.dumps([
                    {
                        "Rules": [
                            {
                                "Resource": ["collection/" + vector_store_name],
                                "Permission": [
                                    "aoss:CreateCollectionItems",
                                    "aoss:DeleteCollectionItems",
                                    "aoss:UpdateCollectionItems",
                                    "aoss:DescribeCollectionItems",
                                ],
                                "ResourceType": "collection",
                            },
                            {
                                "Resource": ["index/" + vector_store_name + "/*"],
                                "Permission": [
                                    "aoss:CreateIndex",
                                    "aoss:DeleteIndex",
                                    "aoss:UpdateIndex",
                                    "aoss:DescribeIndex",
                                    "aoss:ReadDocument",
                                    "aoss:WriteDocument",
                                ],
                                "ResourceType": "index",
                            },
                        ],
                        "Principal": [self.identity],
                        "Description": "Easy data policy",
                    }
                ]),
                type="data",
            )
            print("✅ 데이터 액세스 정책 생성 완료 (재시도)")
        
        return encryption_policy, network_policy, access_policy
    
    def create_oss_collection(self, vector_store_name):
        """
        OpenSearch Serverless 컬렉션 생성
        knowledge_base.py의 create_oss 메서드를 정확히 참고
        
        Args:
            vector_store_name (str): 벡터 스토어(컬렉션) 이름
            
        Returns:
            tuple: (host, collection, collection_id, collection_arn, endpoint)
        """
        print(f"📦 컬렉션 '{vector_store_name}' 생성 중...")
        
        try:
            collection = self.aoss_client.create_collection(
                name=vector_store_name, 
                type="VECTORSEARCH",
                description=f"한국어 자동차 매뉴얼 RAG 시스템용 OpenSearch Serverless 컬렉션 - {vector_store_name}"
            )
            collection_id = collection["createCollectionDetail"]["id"]
            collection_arn = collection["createCollectionDetail"]["arn"]
            print(f"✅ 컬렉션 생성 요청 완료. ID: {collection_id}")
        except self.aoss_client.exceptions.ConflictException:
            print("⚠️ 컬렉션이 이미 존재합니다. 기존 컬렉션을 가져옵니다.")
            collection = self.aoss_client.batch_get_collection(
                names=[vector_store_name]
            )["collectionDetails"][0]
            collection_id = collection["id"]
            collection_arn = collection["arn"]
        
        # OpenSearch serverless 컬렉션 URL 생성 (knowledge_base.py와 동일한 방식)
        host = collection_id + "." + self.region_name + ".aoss.amazonaws.com"
        endpoint = f"https://{host}"
        
        print(f"🔗 컬렉션 호스트: {host}")
        print(f"🔗 컬렉션 엔드포인트: {endpoint}")
        
        # 컬렉션 생성 대기 (knowledge_base.py와 동일한 방식)
        print("⏳ 컬렉션 활성화 대기 중...")
        self.wait_for_collection_active(vector_store_name)
        
        return host, collection, collection_id, collection_arn, endpoint
    
    def wait_for_collection_active(self, vector_store_name):
        """
        컬렉션이 활성화될 때까지 대기
        knowledge_base.py의 대기 로직을 정확히 참고
        
        Args:
            vector_store_name (str): 컬렉션 이름
        """
        # knowledge_base.py와 동일한 방식으로 상태 확인
        response = self.aoss_client.batch_get_collection(names=[vector_store_name])
        
        # 주기적으로 컬렉션 상태 확인 (knowledge_base.py와 동일)
        while (response["collectionDetails"][0]["status"]) == "CREATING":
            print("Creating collection...")
            self.interactive_sleep(30)  # knowledge_base.py와 동일한 대기 시간
            response = self.aoss_client.batch_get_collection(names=[vector_store_name])
        
        print("\n✅ 컬렉션 생성 성공:")
        print(f"   상태: {response['collectionDetails'][0]['status']}")
        print(f"   이름: {response['collectionDetails'][0]['name']}")
        print(f"   ID: {response['collectionDetails'][0]['id']}")
    
    def interactive_sleep(self, seconds):
        """
        knowledge_base.py의 interactive_sleep 함수와 동일
        시각적 피드백과 함께 대기
        
        Args:
            seconds (int): 대기할 초 수
        """
        dots = ""
        for i in range(seconds):
            dots += "."
            print(dots, end="\r")
            time.sleep(1)
    
    def create_collection_with_policies(self, collection_name):
        """
        정책과 함께 컬렉션 생성 (전체 프로세스)
        knowledge_base.py의 create_or_retrieve_knowledge_base 패턴을 참고
        
        Args:
            collection_name (str): 컬렉션 이름
            
        Returns:
            dict: 컬렉션 정보
        """
        print(f"🚀 OpenSearch Serverless 컬렉션 '{collection_name}' 생성 시작...")
        
        # 1. 보안 정책 생성 (knowledge_base.py Step 3)
        print("="*80)
        print(f"Step 1 - OSS 암호화, 네트워크 및 데이터 액세스 정책 생성")
        encryption_policy, network_policy, access_policy = self.create_policies_in_oss(collection_name)
        
        # 2. 컬렉션 생성 (knowledge_base.py Step 4)
        print("="*80)
        print(f"Step 2 - OSS 컬렉션 생성 (몇 분 소요될 수 있습니다)")
        host, collection, collection_id, collection_arn, endpoint = self.create_oss_collection(collection_name)
        
        # 3. 컬렉션 정보 반환
        collection_info = {
            'name': collection_name,
            'id': collection_id,
            'arn': collection_arn,
            'endpoint': endpoint,
            'host': host,
            'status': 'ACTIVE',
            'region': self.region_name,
            'policies': {
                'encryption': encryption_policy['securityPolicyDetail']['name'],
                'network': network_policy['securityPolicyDetail']['name'],
                'access': access_policy['accessPolicyDetail']['name']
            }
        }
        
        return collection_info
    
    def update_network_policy_for_dashboard(self, collection_name: str, network_policy_name: str):
        """
        기존 컬렉션의 네트워크 정책을 업데이트하여 대시보드 액세스 추가
        
        Args:
            collection_name (str): 컬렉션 이름
            network_policy_name (str): 네트워크 정책 이름
        """
        try:
            print(f"🔄 네트워크 정책 업데이트 중: {network_policy_name}")
            
            # 기존 정책 업데이트
            updated_policy = self.aoss_client.update_security_policy(
                name=network_policy_name,
                policyVersion="MTY3NDY2MzA2MzMzNF8x",  # 정책 버전 (실제로는 동적으로 가져와야 함)
                policy=json.dumps([
                    {
                        "Rules": [
                            {
                                "Resource": ["collection/" + collection_name],
                                "ResourceType": "collection",
                            },
                            {
                                "Resource": ["collection/" + collection_name],
                                "ResourceType": "dashboard"
                            }
                        ],
                        "AllowFromPublic": True,
                    }
                ]),
                type="network",
            )
            
            print("✅ 네트워크 정책 업데이트 완료 (OpenSearch 대시보드 액세스 추가)")
            return updated_policy
            
        except Exception as e:
            print(f"❌ 네트워크 정책 업데이트 실패: {e}")
            return None

def main():
    """
    메인 함수 - OpenSearch Serverless 컬렉션 생성 실행
    """
    # 설정값 (us-west-2 리전 사용, 고유한 컬렉션 이름)
    timestamp = str(int(time.time()))[-6:]  # 마지막 6자리
    COLLECTION_NAME = f"rag-car-manual-{timestamp}"  # 고유한 컬렉션 이름
    REGION = "us-west-2"  # us-west-2로 변경
    
    try:
        # OpenSearch Serverless 관리자 생성
        manager = OpenSearchServerlessManager(region_name=REGION)
        
        print(f"📝 생성할 컬렉션: {COLLECTION_NAME}")
        
        # 사용자 선택
        print("\n🔧 작업을 선택하세요:")
        print("1. 새 컬렉션 생성 (대시보드 액세스 포함)")
        print("2. 기존 컬렉션의 네트워크 정책 업데이트")
        
        choice = input("선택 (1 또는 2): ").strip()
        
        if choice == "1":
            # 컬렉션 생성
            collection_info = manager.create_collection_with_policies(COLLECTION_NAME)
            
            # 결과 출력
            print("\n" + "="*60)
            print("🎉 OpenSearch Serverless 컬렉션 생성 완료!")
            print("="*60)
            print(f"컬렉션 이름: {collection_info['name']}")
            print(f"컬렉션 ID: {collection_info['id']}")
            print(f"상태: {collection_info['status']}")
            print(f"엔드포인트: {collection_info['endpoint']}")
            print(f"ARN: {collection_info['arn']}")
            print(f"리전: {collection_info['region']}")
            print("정책:")
            print(f"  - 암호화: {collection_info['policies']['encryption']}")
            print(f"  - 네트워크: {collection_info['policies']['network']} (대시보드 액세스 포함)")
            print(f"  - 액세스: {collection_info['policies']['access']}")
            print("="*60)
            
            # 설정 파일 저장 (다음 단계에서 사용)
            config = {
                'collection_name': collection_info['name'],
                'collection_id': collection_info['id'],
                'endpoint': collection_info['endpoint'],
                'host': collection_info['host'],
                'arn': collection_info['arn'],
                'region': collection_info['region'],
                'policies': collection_info['policies']
            }
            
            with open('opensearch_config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print("💾 설정 파일 저장 완료: opensearch_config.json")
            
        elif choice == "2":
            # 기존 컬렉션 업데이트
            existing_collection = input("기존 컬렉션 이름을 입력하세요 (예: rag-car-manual-208167): ").strip()
            network_policy_name = input("네트워크 정책 이름을 입력하세요: ").strip()
            
            if existing_collection and network_policy_name:
                result = manager.update_network_policy_for_dashboard(existing_collection, network_policy_name)
                if result:
                    print(f"\n✅ 컬렉션 '{existing_collection}'의 네트워크 정책이 업데이트되었습니다.")
                    print("🖥️ 이제 OpenSearch 대시보드에 액세스할 수 있습니다.")
                else:
                    print("❌ 네트워크 정책 업데이트에 실패했습니다.")
            else:
                print("❌ 컬렉션 이름과 정책 이름을 모두 입력해야 합니다.")
        else:
            print("❌ 잘못된 선택입니다. 1 또는 2를 입력하세요.")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
