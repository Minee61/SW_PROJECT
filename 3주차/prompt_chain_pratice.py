query = rewrite_chain.run(user_input)
# LangChain의 rewrite_chain / LLM을 호출해서 더 검색하기 적합한 문장으로 변환 
# 검색을 정확도를 높이기 위해 자연어를 검색 용도에 최적화된 문장으로 바꾸기. 

from neo4j import GraphDatabase 
kg_context = search_kg_with_query(query)
# Neo4j에 연결 / 위에 저 query를 기반으로 KG에 트리플을 검색 
# search_kg_with_query 이건 그냥 함수 

'''
def search_kg_with_query(query_text):
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password")))

    cypher = f"""
    MATHCH (s)-[r]->(O)
    WHERE s.name CONTAINS '{query_text}' OR o.name CONTAINS '{query_text}'
    RETURN s.name AS subject, typer(r) AS realtion, o.name AS object 
    LIMIT 5 """

    with driver.session() as session:
        results = session.run(cypher)
        triples = [f"{r['subject']} - [{rp'relation']}]-> {r['object']}" for r in results]

        return "\n".join(triples)
'''
# 출력 예시 
# "지식 그래프 -[보완하다]-> 언어 모델 
#  지식 그래프 -[제공한다]-> 구조화 지식
#  언어 모델 -[요구한다]-> 외부지식"

# 사용이유?
# llm은 정확하고 사실적인 정보 보장 X 
# KG에서 구조화된 지식 찾아서 주면 llm의 답이 더 정확해짐 
