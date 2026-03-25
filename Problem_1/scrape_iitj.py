import requests
from bs4 import BeautifulSoup
import PyPDF2
import os

def extract_text_from_url(url):
    """Scrapes paragraph text from a given webpage."""
    print(f"Scraping Web URL: {url}")
    try:
        # We use a user-agent so the university server doesn't block us as a bot
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text only from paragraph tags to avoid navigation menus and raw code
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text(strip=True) for p in paragraphs])
        return text + "\n\n"
    except Exception as e:
        print(f"  -> Failed to scrape {url}: {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    """Extracts text from a locally saved PDF file."""
    print(f"Extracting PDF: {pdf_path}")
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text + "\n\n"
    except Exception as e:
        print(f"  -> Failed to read PDF {pdf_path}: {e}")
        return ""

def main():
    # 1. Put the URLs you want to scrape here
    urls_to_scrape = [
        "https://www.iitj.ac.in/", 
        "https://www.iitj.ac.in/Computer-Science-Engineering/en/Artificial-Intelligence-and-Machine-Learning",
        "https://www.iitj.ac.in/Computer-Science-Engineering/en/Systems-Software",
        "https://www.iitj.ac.in/mathematics/",
        "https://www.iitj.ac.in/computer-science-engineering/" # CSE Department
    ]
    
    # 2. Put the names of the PDFs you downloaded here
    pdfs_to_extract = [
        "Faculty.pdf",
        "P.h.D.pdf",
        "regulations.pdf",
    ]

    master_corpus = ""

    # Process URLs
    for url in urls_to_scrape:
        master_corpus += extract_text_from_url(url)

    # Process PDFs
    for pdf in pdfs_to_extract:
        if os.path.exists(pdf):
            master_corpus += extract_text_from_pdf(pdf)
        else:
            print(f"Warning: Could not find file {pdf}. Did you download it?")

    # Save to the master file
    with open("iitj_raw_corpus.txt", "w", encoding="utf-8") as f:
        f.write(master_corpus)
        
    print(f"\nSuccess! Extracted {len(master_corpus.split())} total raw words.")
    print("Saved to 'iitj_raw_corpus.txt'. You can now run the Task 1 cleaning script!")

if __name__ == "__main__":
    main()
