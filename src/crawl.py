import os
import re
import json
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, RecursiveUrlLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
import easyocr
from dotenv import load_dotenv
import numpy as np
from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from time import sleep
from langchain.schema import Document



load_dotenv()

def bs4_extractor(html: str) -> str:
    """
    Hàm trích xuất và làm sạch nội dung từ HTML
    Args:
        html: Chuỗi HTML cần xử lý
    Returns:
        str: Văn bản đã được làm sạch, loại bỏ các thẻ HTML và khoảng trắng thừa
    """
    soup = BeautifulSoup(html, "html.parser")  # Phân tích cú pháp HTML
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()  # Xóa khoảng trắng và dòng trống thừa

def selenium_extractor(url):
    """
    Sử dụng Selenium để crawl dữ liệu từ URL có JavaScript và cookies
    Args:
        url (str): URL của trang web
    Returns:
        str: Nội dung HTML đã được tải từ trang web
    """
    # Set up the Selenium WebDriver (you can also use Firefox or other browsers)
    options = Options()
    options.headless = True  # Run browser in headless mode (without GUI)
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    # Wait for the page to load (adjust the time based on the website)
    sleep(5)  # Wait for content to load

    # Now get the page content
    page_source = driver.page_source
    driver.quit()

    return page_source

def crawl_web(url_data):
    """
    Hàm crawl dữ liệu từ URL với chế độ đệ quy
    Args:
        url_data (str): URL gốc để bắt đầu crawl
    Returns:
        list: Danh sách các Document object, mỗi object chứa nội dung đã được chia nhỏ
              và metadata tương ứng
    """
    # Tạo loader với độ sâu tối đa là 4 cấp
    loader = RecursiveUrlLoader(url=url_data, extractor=bs4_extractor, max_depth=4)
    docs = loader.load()  # Tải nội dung
    print('length: ', len(docs))  # In số lượng tài liệu đã tải
    
    # Chia nhỏ văn bản thành các đoạn 10000 ký tự, với 500 ký tự chồng lấp
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    all_splits = text_splitter.split_documents(docs)
    print('length_all_splits: ', len(all_splits))  # In số lượng đoạn văn bản sau khi chia
    return all_splits

def web_base_loader(url_data):
    """
    Hàm tải dữ liệu từ một URL đơn (không đệ quy)
    Args:
        url_data (str): URL cần tải dữ liệu
    Returns:
        list: Danh sách các Document object đã được chia nhỏ
    """
    loader = WebBaseLoader(url_data)
    docs = loader.load()
    print('length: ', len(docs))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    all_splits = text_splitter.split_documents(docs)
    return all_splits

def extract_text_from_pdf(pdf_path):
    """
    Trích xuất nội dung văn bản từ file PDF
    Args:
        pdf_path (str): Đường dẫn đến file PDF
    Returns:
        str: Toàn bộ nội dung văn bản từ file PDF
    """

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def ocr_extract_from_pdf(pdf_path):
    """
    Sử dụng OCR để trích xuất nội dung từ file PDF dạng hình ảnh với EasyOCR
    Args:
        pdf_path (str): Đường dẫn đến file PDF
    Returns:
        str: Toàn bộ nội dung văn bản được trích xuất từ các trang PDF
    """
    images = convert_from_path(pdf_path)
    # reader = easyocr.Reader(["en"], gpu=True)  # Kích hoạt GPU
    reader = easyocr.Reader(["vi", "en"], gpu=True)  # Hỗ trợ tiếng Việt và tiếng Anh
    all_text = ""
    for i, image in enumerate(images):
        print(f"Processing page {i + 1} with OCR...")
        # Chuyển đổi PIL Image thành numpy array
        image_np = np.array(image)
        text = reader.readtext(image_np, detail=0)  # Trích xuất văn bản
        all_text += "\n".join(text) + "\n"
    return all_text

def process_pdf(pdf_path, use_ocr=False):
    """
    Xử lý file PDF để chia nhỏ nội dung và phục vụ RAG
    Args:
        pdf_path (str): Đường dẫn đến file PDF
        use_ocr (bool): Sử dụng OCR nếu PDF không chứa văn bản gốc
    Returns:
        list: Danh sách các Document object đã được chia nhỏ
    """
    if use_ocr:
        pdf_text = ocr_extract_from_pdf(pdf_path)
    else:
        documents = extract_text_from_pdf(pdf_path)  # Returns a list of Document objects
        pdf_text = "\n".join([doc.page_content for doc in documents])  # Combine all text

    # Chia nhỏ văn bản thành các đoạn
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    documents = text_splitter.split_text(pdf_text)

    # Chuyển đổi thành định dạng Document với metadata đơn giản
    docs = [
        Document(page_content=chunk, metadata={"source": os.path.basename(pdf_path)})
        for chunk in documents
    ]
    print(f"Processed {len(docs)} chunks from PDF: {pdf_path}")
    return docs

def save_data_locally(documents, filename, directory):
    """
    Lưu danh sách documents vào file JSON
    Args:
        documents (list): Danh sách các Document object cần lưu
        filename (str): Tên file JSON (ví dụ: 'data.json')
        directory (str): Đường dẫn thư mục lưu file
    Returns:
        None: Hàm không trả về giá trị, chỉ lưu file và in thông báo
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)

    data_to_save = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in documents]
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data_to_save, file, indent=4, ensure_ascii=False)
    print(f'Data saved to {file_path}')

def main():
    """
    Hàm chính điều khiển luồng chương trình:
    1. Crawl dữ liệu từ trang web
    2. Xử lý file PDF
    3. Lưu dữ liệu vào file JSON
    """
    # Crawl dữ liệu từ web
    # web_data = crawl_web('https://www.stack-ai.com/docs')
    # save_data_locally(web_data, 'web_data.json', 'data')

    # Xử lý file PDF (thử với OCR nếu cần)
    pdf_data = process_pdf('Nghị định quy định xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ và đường sắt do Bộ Giao thông vận tải ban hành (Ban hành 2020).pdf', use_ocr=False)
    save_data_locally(pdf_data, 'Nghị định quy định xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ và đường sắt do Bộ Giao thông vận tải ban hành (Ban hành 2020)', 'data')

if __name__ == "__main__":
    main()