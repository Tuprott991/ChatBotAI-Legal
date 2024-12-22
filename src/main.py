"""
File chính để chạy ứng dụng Chatbot AI
Chức năng: 
- Tạo giao diện web với Streamlit
- Xử lý tương tác chat với người dùng
- Kết nối với AI model để trả lời
"""

# === IMPORT CÁC THƯ VIỆN CẦN THIẾT ===
import streamlit as st  # Thư viện tạo giao diện web
from dotenv import load_dotenv  # Đọc file .env chứa API key
from database_module import seed_milvus, seed_milvus_live  # Hàm xử lý dữ liệu
from agent import get_retriever as get_openai_retriever, get_llm_and_agent as get_openai_agent
from ollama_agent import get_retriever as get_ollama_retriever, get_llm_and_agent as get_ollama_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from my_speech import transcribe_audio
from audio_recorder_streamlit import audio_recorder
import random
import base64

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


# def set_background(png_file):
#     bin_str = get_base64(png_file)
#     page_bg_img = '''
#     <style>
#     .sidebar {
#     background-image: url("data:image/png;base64,%s");
#     background-size: cover;
#     opacity: 1;
#     }
#     </style>
#     ''' % bin_str
#     st.markdown(page_bg_img, unsafe_allow_html=True)

# === THIẾT LẬP GIAO DIỆN TRANG WEB ===
def setup_page():
    """
    Cấu hình trang web cơ bản
    """
    st.set_page_config(
        page_title="VinhHien ChatBot",  # Tiêu đề tab trình duyệt
        page_icon="🤖",  # Icon tab``
        layout="wide"  # Giao diện rộng
    )
    # set_background('thcsvinhhien.jpg')  # Thiết lập hình nền

# === KHỞI TẠO ỨNG DỤNG ===
def initialize_app():
    """
    Khởi tạo các cài đặt cần thiết:
    - Đọc file .env chứa API key
    - Cấu hình trang web
    """
    load_dotenv()  # Đọc API key từ file .env
    setup_page()  # Thiết lập giao diện

# === THANH CÔNG CỤ BÊN TRÁI ===
def setup_sidebar():
    """
    Tạo thanh công cụ bên trái với các tùy chọn
    """
    with st.sidebar:
        st.title("⚙️ Cấu hình")
        

        # Phần 1: Chọn Model để trả lời
        st.header("🤖 Model AI")
        model_choice = st.radio(
            "Chọn AI Model để trả lời:",
            ["OpenAI GPT-4", "xAI Grok (free API)", "Meta LLama3.1 (Ollama - Local)"]
        )

        # Phần 2: Chọn lĩnh vực 

        # Thêm phần chọn collection để query
        st.header("🔍 Chọn lĩnh vực pháp luật")
        collections = ["Luật giao thông đường bộ", "Luật giao thông đường bộ (BGE-M3)", "Luật lao động", "Luật hôn nhân và gia đình", "Luật đất đai"]

        collection_to_query = st.selectbox(
            "Chọn lĩnh vực pháp luật để truy vấn:",
            collections,
            help="Chọn lĩnh vực pháp luật bạn muốn sử dụng để tìm kiếm thông tin"
        )

        if collection_to_query == "Luật giao thông đường bộ":
            collection_to_query = "LGTDB"
        elif collection_to_query == "Luật giao thông đường bộ (BGE-M3)":
            collection_to_query = "LGTDB_BGE"

        # Phần 3: Chọn Embeddings Model
        st.header("🔤 Embeddings Model")
        embeddings_choice = st.radio(
            "Chọn Embeddings Model:",
            ["OpenAI", "Ollama"]
        )
        use_ollama_embeddings = (embeddings_choice == "Ollama")
        
        # Phần 4: Cấu hình Data
        st.header("📚 Nạp thêm dữ liệu")
        data_source = st.radio(
            "Chọn nguồn dữ liệu:",
            ["File Local", "URL trực tiếp"]
            
        )
        # Xử lý nguồn dữ liệu dựa trên embeddings đã chọn
        if data_source == "File Local":
            handle_local_file(use_ollama_embeddings)
        else:
            handle_url_input(use_ollama_embeddings)
            
        return model_choice, collection_to_query

def handle_local_file(use_ollama_embeddings: bool):
    """
    Xử lý khi người dùng chọn tải file
    """
    collection_name = st.text_input(
        "Tên collection trong Milvus:", 
        "data_test",
        help="Nhập tên collection bạn muốn lưu trong Milvus"
    )
    filename = st.text_input("Tên file JSON:", "stack.json")
    directory = st.text_input("Thư mục chứa file:", "data")
    
    if st.button("Tải dữ liệu từ file"):
        if not collection_name:
            st.error("Vui lòng nhập tên collection!")
            return
            
        with st.spinner("Đang tải dữ liệu..."):
            try:
                seed_milvus(
                    'http://localhost:19530', 
                    collection_name, 
                    filename, 
                    directory, 
                    use_ollama=use_ollama_embeddings
                )
                st.success(f"Đã tải dữ liệu thành công vào collection '{collection_name}'!")
            except Exception as e:
                st.error(f"Lỗi khi tải dữ liệu: {str(e)}")

def handle_url_input(use_ollama_embeddings: bool):
    """
    Xử lý khi người dùng chọn crawl URL
    """
    collection_name = st.text_input(
        "Tên collection trong Milvus:", 
        "data_test_live",
        help="Nhập tên collection bạn muốn lưu trong Milvus"
    )
    url = st.text_input("Nhập URL:", "https://www.stack-ai.com/docs")
    
    if st.button("Crawl dữ liệu"):
        if not collection_name:
            st.error("Vui lòng nhập tên collection!")
            return
            
        with st.spinner("Đang crawl dữ liệu..."):
            try:
                seed_milvus_live(
                    url, 
                    'http://localhost:19530', 
                    collection_name, 
                    'stack-ai', 
                    use_ollama=use_ollama_embeddings
                )
                st.success(f"Đã crawl dữ liệu thành công vào collection '{collection_name}'!")
            except Exception as e:
                st.error(f"Lỗi khi crawl dữ liệu: {str(e)}")


# === GIAO DIỆN CHAT CHÍNH ===
def setup_chat_interface(model_choice):
    st.title("🤖💬 Legal Assistant AI ChatBot")
    
    # Caption động theo model
    if model_choice == "OpenAI GPT-4":
        st.caption("✨ Trợ lý AI pháp luật được thực hiện bởi nhóm học sinh trường THCS Vinh Hiền hỗ trợ bởi framework LangChain")
    elif model_choice == "xAI Grok (free API)":
        st.caption("✨ Trợ lý AI pháp luật được thực hiện bởi nhóm học sinh trường THCS Vinh Hiền hỗ trợ bởi framework LangChain")
    else:
        st.caption("✨ Trợ lý AI được thực hiện bởi nhóm học sinh trường THCS Vinh Hiền hỗ trợ bởi framework LangChain và Open Source LLM Ollama LLaMA3.1")
    
    # Khởi tạo bộ nhớ chat
    msgs = StreamlitChatMessageHistory(key="langchain_messages") # Đúng thì chỗ này nên lưu chat history vào trong backend database
    
    # Tạo đoạn chat mới khi bắt đầu 
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Chào bạn, tôi có thể giúp gì cho bạn?"}
        ]
        msgs.add_ai_message("Tôi có thể giúp gì cho bạn?")

    # hiển thị lịch sử chat
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

    return msgs


# === XỬ LÝ TIN NHẮN NGƯỜI DÙNG ===
def handle_user_input(msgs, agent_executor):
    """
    Xử lý khi người dùng gửi tin nhắn:
    1. Hiển thị tin nhắn người dùng
    2. Gọi AI xử lý và trả lời
    3. Lưu vào lịch sử chat
    """
    if prompt := st.chat_input("Hãy hỏi tôi đi nào!!!" ):
        # Lưu và hiển thị tin nhắn người dùng

        st.session_state.messages.append({"role": "human", "content": prompt}) #session_state là quản lý cuộc hội thoại đó
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

        # Xử lý và hiển thị câu trả lời
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            
            # Lấy lịch sử chat
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]
            ]

            # Gọi AI xử lý
            response = agent_executor.invoke(
                {
                    "input": prompt,
                    "chat_history": chat_history
                },
                {"callbacks": [st_callback]} # AI gen ra chữ nào thì chỗ này sẽ hứng lại và trả cho user 
            )

            # Lưu và hiển thị câu trả lời
            output = response["output"] # Lấy output
            st.session_state.messages.append({"role": "assistant", "content": output}) # Thêm vào trong session state
            msgs.add_ai_message(output) # Lưu vào lịch sử hội thoại
            st.write(output) # Ghi lên màn hình


def handle_user_input_with_microphone(msgs, agent_executor):
    """
    Xử lý khi người dùng gửi tin nhắn qua thanh chat hoặc sử dụng Microphone:
    Thanh chat input và nút micro nằm liền kề nhau, không cần chia cột.
    """
    # Chat input: cho phép gõ text 


    user_input = st.chat_input(placeholder="Hãy hỏi tôi bằng cách nhập hoặc nhấn 🎤 để nói...")

    audio_bytes = audio_recorder(
        text="", 
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="2x"
    )

    # Xử lý đầu vào từ microphone
    if audio_bytes:
        with st.spinner("🎙️ Đang nhận diện giọng nói..."):
            # Chạy nhận diện âm thanh qua microphone
            text_from_audio = transcribe_audio(audio_bytes)
            if text_from_audio.startswith("Lỗi"):
                # Thông báo lỗi nếu không nhận diện được
                # st.error(text_from_audio)
                pass
            else:
                # Khi nhận diện thành công -> điền sẵn vào input box
                # st.success(f"📥 Nhận diện: {text_from_audio}")
                st.session_state.user_temp_input = text_from_audio

    # Nếu có đầu vào (từ thanh chat hoặc từ giọng nói)
    if user_input or st.session_state.get("user_temp_input"):
        # Lấy prompt từ thanh chat hoặc từ giọng nói (ưu tiên thanh chat)
        prompt = user_input or st.session_state.user_temp_input
        st.session_state.user_temp_input = ""  # Reset header của microphone input

        # Lưu tin nhắn người dùng và hiển thị
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

        # Xử lý dữ liệu với AI
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            
            # Lấy lịch sử hội thoại già hơn
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]
            ]

            # ALGORITHM GPT processing
            response = agent_executor.invoke(
                {
                    "input": prompt,
                    "chat_history": chat_history
                },
                {"callbacks": [st_callback]}  # Callback để stream output lên giao diện
            )

            # Hiển thị đáp án của AI
            output = response["output"]
            st.session_state.messages.append({"role": "assistant", "content": output})
            msgs.add_ai_message(output)
            st.write(output)

# === HÀM CHÍNH ===
def main():
    """
    Hàm chính điều khiển luồng chương trình
    """
    initialize_app()
    model_choice, collection_to_query = setup_sidebar()
    msgs = setup_chat_interface(model_choice)
    
    # Khởi tạo AI dựa trên lựa chọn model để trả lời
    if model_choice == "OpenAI GPT-4":
        retriever = get_openai_retriever(collection_to_query)
        print(collection_to_query)
        agent_executor = get_openai_agent(retriever, "gpt4")
    elif model_choice == "OpenAI Grok":
        retriever = get_openai_retriever(collection_to_query)
        agent_executor = get_openai_agent(retriever, "grok")
    else:
        retriever = get_ollama_retriever(collection_to_query, use_ollama=True)
        print(collection_to_query)
        agent_executor = get_ollama_agent(retriever)

    handle_user_input_with_microphone(msgs, agent_executor)\
    


# Chạy ứng dụng
if __name__ == "__main__":
    main() 