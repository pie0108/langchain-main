import streamlit as st  # Streamlit은 간단한 웹 애플리케이션을 만들 수 있는 Python 라이브러리입니다.
import tiktoken  # tiktoken은 텍스트를 토큰으로 변환하는 도구입니다.
from loguru import logger  # loguru는 로그를 기록할 때 사용하는 라이브러리입니다.

# langchain 패키지에서 필요한 클래스와 함수들을 가져옵니다.
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

# 메인 함수입니다. 프로그램이 실행될 때 가장 먼저 호출됩니다.
def main():
    # 웹 페이지의 제목과 아이콘을 설정합니다.
    st.set_page_config(
    page_title="DirChat",
    page_icon=":books:")

    # 웹 페이지에 보여줄 제목을 설정합니다.
    st.title("_DAESUNG DATA :red[대성에너지]_ :books:")

    # 상태 저장을 위한 변수를 초기화합니다. 세션 상태는 사용자가 페이지를 새로 고침해도 데이터를 유지할 수 있게 합니다.
    if "conversation" not in st.session_state:
        st.session_state.conversation = None  # 대화를 저장할 변수입니다.

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None  # 채팅 기록을 저장할 변수입니다.

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None  # 프로세스 완료 여부를 저장할 변수입니다.

    # 사이드바에 파일 업로드와 API 키 입력 필드를 만듭니다.
    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file", type=['pdf','docx'], accept_multiple_files=True)  # 파일 업로드
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")  # OpenAI API 키 입력
        process = st.button("Process")  # "Process" 버튼을 만듭니다.
    
    # 사용자가 "Process" 버튼을 눌렀을 때 실행되는 코드입니다.
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()  # API 키가 없으면 메시지를 표시하고 프로그램을 멈춥니다.
        
        # 업로드된 파일에서 텍스트를 추출합니다.
        files_text = get_text(uploaded_files)
        # 텍스트를 작은 조각들로 나눕니다.
        text_chunks = get_text_chunks(files_text)
        # 텍스트 조각들을 벡터로 변환하여 검색이 가능하게 합니다.
        vetorestore = get_vectorstore(text_chunks)
        
        # 대화 체인을 만듭니다. 이 체인은 사용자의 질문에 답변하기 위해 사용됩니다.
        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key) 
        st.session_state.processComplete = True  # 프로세스가 완료되었음을 표시합니다.

    # 대화 메시지를 저장할 변수를 초기화합니다.
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                         "content": "안녕하세요! 대성에너지 GPT 입니다. 궁금하신 것이 있으시면 물어봐주세요!"}]
    
    # 저장된 메시지를 화면에 표시합니다.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):  # 메시지의 역할에 따라 다르게 표시합니다.
            st.markdown(message["content"])

    # 채팅 메시지 기록을 관리하기 위한 객체를 만듭니다.
    history = StreamlitChatMessageHistory(key="chat_messages")

    # 사용자가 질문을 입력하면 실행되는 코드입니다.
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})  # 사용자의 질문을 메시지 리스트에 추가합니다.

        with st.chat_message("user"):
            st.markdown(query)  # 사용자의 질문을 화면에 표시합니다.

        with st.chat_message("assistant"):
            chain = st.session_state.conversation  # 대화 체인을 가져옵니다.

            with st.spinner("Thinking..."):  # 응답을 생성하는 동안 스피너를 표시합니다.
                result = chain({"question": query})  # 질문을 대화 체인에 전달하여 응답을 생성합니다.
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']  # 대화의 기록을 저장합니다.
                response = result['answer']  # 생성된 응답을 저장합니다.
                source_documents = result['source_documents']  # 응답에 사용된 문서들을 가져옵니다.

                st.markdown(response)  # 응답을 화면에 표시합니다.
                with st.expander("참고 문서 확인"):  # 참고 문서를 볼 수 있는 버튼을 만듭니다.
                    st.markdown(source_documents[0].metadata['source'], help=source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help=source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help=source_documents[2].page_content)
                    
        # 생성된 응답을 메시지 리스트에 추가합니다.
        st.session_state.messages.append({"role": "assistant", "content": response})

# 텍스트를 토큰으로 변환하여 길이를 계산하는 함수입니다.
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

# 업로드된 파일에서 텍스트를 추출하는 함수입니다.
def get_text(docs):
    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # 파일 이름을 가져옵니다.
        with open(file_name, "wb") as file:  # 파일을 로컬에 저장합니다.
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        
        # 파일의 형식에 따라 다른 로더를 사용하여 텍스트를 추출합니다.
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)  # 추출된 문서를 리스트에 추가합니다.
    
    return doc_list

# 텍스트를 작은 조각들로 나누는 함수입니다.
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,  # 각 조각의 최대 길이입니다.
        chunk_overlap=100,  # 조각 간의 겹침 부분입니다.
        length_function=tiktoken_len  # 텍스트 길이를 계산하는 함수입니다.
    )
    chunks = text_splitter.split_documents(text)  # 문서를 조각으로 나눕니다.
    return chunks

# 텍스트 조각들을 벡터로 변환하여 검색이 가능하게 하는 함수입니다.
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",  # 벡터화에 사용할 모델입니다.
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)  # 텍스트 조각을 벡터로 변환합니다.
    return vectordb

# 대화 체인을 생성하는 함수입니다. 이 체인은 질문에 답변할 때 사용됩니다.
def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)  # OpenAI 모델을 사용합니다.
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type='mmr', vervose=True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose=True
        )
    return conversation_chain  # 생성된 대화 체인을 반환합니다.

# 프로그램이 실행될 때 가장 먼저 호출되는 메인 함수입니다.
if __name__ == '__main__':
    main()



# 코드 설명

# 라이브러리 임포트: 코드의 첫 부분에서는 프로그램에서 사용할 다양한 라이브러리를 불러옵니다. Streamlit은 간단한 웹 애플리케이션을 만들 수 있게 해주고, tiktoken은 텍스트를 토큰으로 변환하며, loguru는 로그를 기록합니다. langchain 패키지에서는 자연어 처리와 관련된 다양한 도구들을 가져옵니다.

# 메인 함수: main() 함수는 프로그램이 실행될 때 가장 먼저 호출되는 함수로, 여기에서 웹 페이지의 제목 설정, 파일 업로드, API 키 입력, 그리고 대화 체인의 생성과 같은 주요 기능이 실행됩니다.

# 파일 업로드: 사용자는 PDF, DOCX, PPTX 파일을 업로드할 수 있으며, 업로드된 파일에서 텍스트를 추출하여 작은 조각들로 나누고, 이를 벡터화하여 검색할 수 있게 합니다.

# 대화 기능: 사용자는 질문을 입력할 수 있고, 프로그램은 미리 생성된 대화 체인을 통해 이 질문에 대한 답변을 생성합니다. 답변에 사용된 참고 문서도 확인할 수 있습니다.

# 세션 상태: 프로그램이 실행되는 동안 데이터(예: 대화 기록, 처리 완료 여부)를 유지하기 위해 Streamlit의 세션 상태를 사용합니다.

# 함수: get_text(), get_text_chunks(), get_vectorstore(), get_conversation_chain()와 같은 함수들은 각각 파일에서 텍스트를 추출하고, 텍스트를 조각으로 나누며, 벡터화를 통해 검색이 가능하게 하며, 대화 체인을 생성하는 역할을 합니다.
