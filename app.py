import pandas as pd
import streamlit as st

from src.analysis import category_counts, extract_issue_keywords
from src.config import INDEX_DATA_PATH, INDEX_PATH, default_case_data_path
from src.data import load_cases
from src.embedding import LegalEmbedder, resolve_model_name
from src.search import build_index, load_index, semantic_search

st.set_page_config(
    page_title="유사 판례 검색",
    page_icon="⚖️",
    layout="wide",
)


@st.cache_resource
def get_embedder() -> LegalEmbedder:
    return LegalEmbedder()


@st.cache_data
def get_index() -> tuple[pd.DataFrame, object]:
    return load_index()


def rebuild_index(cases: pd.DataFrame) -> None:
    with st.spinner("판례 임베딩을 생성하고 있습니다..."):
        build_index(cases, get_embedder())
    get_index.clear()


st.title("유사 판례 검색 및 쟁점 분석")
st.caption("Sentence-BERT 임베딩과 코사인 유사도를 이용한 교육용 검색 시스템")

with st.sidebar:
    st.header("데이터 및 모델")
    st.text(f"모델: {resolve_model_name()}")
    uploaded_file = st.file_uploader("판례 CSV 업로드", type=["csv"])
    source_cases = None
    try:
        source_cases = load_cases(uploaded_file or default_case_data_path())
        st.success(f"데이터 {len(source_cases):,}건 확인")
    except ValueError as error:
        st.error(str(error))

    if st.button("인덱스 생성", type="primary", use_container_width=True):
        if source_cases is not None:
            rebuild_index(source_cases)
            st.success("검색 인덱스 생성 완료")

    index_ready = INDEX_PATH.exists() and INDEX_DATA_PATH.exists()
    st.info("인덱스 준비됨" if index_ready else "인덱스 생성이 필요합니다.")

query = st.text_area(
    "사건 내용 또는 법률 쟁점",
    placeholder="예: 온라인 쇼핑몰에서 구매한 제품의 하자로 환불을 요청했으나 판매자가 거절하였다.",
    height=130,
)

col1, col2 = st.columns([3, 1])
with col1:
    search_clicked = st.button("유사 판례 검색", type="primary", use_container_width=True)
with col2:
    top_k = st.selectbox("검색 개수", [3, 5, 10], index=1)

if search_clicked:
    if not index_ready:
        st.warning("사이드바에서 먼저 인덱스를 생성하세요.")
    elif not query.strip():
        st.warning("검색할 내용을 입력하세요.")
    else:
        try:
            cases, embeddings = get_index()
            with st.spinner("의미가 유사한 판례를 검색하고 있습니다..."):
                results = semantic_search(
                    query, cases, embeddings, get_embedder(), top_k=top_k
                )

            st.subheader("검색 결과")
            for rank, row in results.iterrows():
                title = (
                    f"{rank + 1}. {row['case_name']} "
                    f"· 유사도 {row['similarity']:.1%}"
                )
                with st.expander(title, expanded=rank == 0):
                    st.write(
                        f"**법원/선고일:** {row['court']} / {row['decision_date']}"
                    )
                    st.write(f"**사건번호:** {row['case_number']}")
                    st.write(f"**분야:** {row['category']}")
                    st.write(f"**핵심 쟁점:** {row['issues']}")
                    st.write(f"**판결 요약:** {row['summary']}")
                    if row["source_url"]:
                        st.link_button("원문 확인", row["source_url"])

            st.subheader("쟁점 분석")
            analysis_left, analysis_right = st.columns(2)
            with analysis_left:
                st.write("**공통 쟁점 키워드**")
                keywords = extract_issue_keywords(results)
                st.write(", ".join(word for word, _ in keywords) or "추출 결과 없음")
            with analysis_right:
                st.write("**사건 분야 분포**")
                st.bar_chart(category_counts(results), x="category", y="count")
        except (ValueError, FileNotFoundError) as error:
            st.error(str(error))

st.divider()
st.caption(
    "교육·연구용 프로젝트입니다. 검색 결과는 법률 자문이 아니며 실제 판례 원문을 확인해야 합니다."
)
