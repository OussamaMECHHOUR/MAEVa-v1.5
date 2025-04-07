import argparse
import os
from time import sleep

import requests
from lxml import etree

DEFAULT_INPUT_PDF_DIR = "outputs/Core API"
DEFAULT_XML_DIR = "outputs/GROBID/XML-TEI files"
DEFAULT_TXT_DIR = "outputs/GROBID/TXT files"
DEFAULT_CORPUS_FILE = "datasets/corpora/Corpus (Core).txt"
DEFAULT_GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

def send_pdf_to_grobid(pdf_path, output_path, grobid_url):
    try:
        with open(pdf_path, 'rb') as pdf_file:
            response = requests.post(
                grobid_url,
                files={'input': pdf_file},
                headers={'Accept': 'application/xml'}
            )
        if response.status_code == 200:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"[GROBID OK] {os.path.basename(pdf_path)}")
            return True
        else:
            print(f"[GROBID FAIL] {pdf_path} ({response.status_code})")
            return False
    except Exception as e:
        print(f"[GROBID ERROR] {pdf_path}: {e}")
        return False

def process_pdf_directory(pdf_dir, xml_dir, grobid_url):
    os.makedirs(xml_dir, exist_ok=True)
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            xml_path = os.path.join(xml_dir, os.path.splitext(filename)[0] + '.xml')
            success = send_pdf_to_grobid(pdf_path, xml_path, grobid_url)
            if not success:
                try:
                    os.remove(pdf_path)
                    print(f"[REMOVED] Failed PDF: {pdf_path}")
                except Exception as e:
                    print(f"[DELETE ERROR] {pdf_path}: {e}")
            sleep(1)

def process_xml_file(xml_path, txt_path):
    try:
        tree = etree.parse(xml_path)
        root = tree.getroot()
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        body = root.find(".//tei:body", namespaces=ns)
        if body is None:
            print(f"[NO BODY] {xml_path}")
            return

        for head in body.findall(".//tei:head", namespaces=ns):
            head.getparent().remove(head)

        unwanted_tags = [
            "tei:div[@type='acknowledgment']", "tei:div[@type='availability']",
            "tei:note[@place='headnote']", "tei:note[@place='footnote']",
            "tei:listBibl", "tei:page", "tei:div[@type='toc']", "tei:div[@type='funding']",
            "tei:div[@type='annex']", "tei:titlePage", "tei:front",
            "tei:figure[@type='table']", "tei:figure", "tei:figure[@type='box']",
            "tei:ref[@type='biblio']", "tei:ref[@type='bibr']", "tei:ref[@type='url']",
        ]
        for tag in unwanted_tags:
            for el in body.findall(f".//{tag}", namespaces=ns):
                el.getparent().remove(el)

        text = ' '.join(body.itertext()).strip()
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[TEXT OK] {os.path.basename(txt_path)}")
    except Exception as e:
        print(f"[XML ERROR] {xml_path}: {e}")

def process_xml_directory(xml_dir, txt_dir):
    os.makedirs(txt_dir, exist_ok=True)
    for filename in os.listdir(xml_dir):
        if filename.lower().endswith('.xml'):
            xml_path = os.path.join(xml_dir, filename)
            txt_path = os.path.join(txt_dir, os.path.splitext(filename)[0] + '.txt')
            process_xml_file(xml_path, txt_path)

def concatenate_txt_files(input_dir, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(input_dir):
            if filename.lower().endswith('.txt'):
                file_path = os.path.join(input_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read().strip() + "\n")
    print(f"\nFinal corpus saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process PDF articles into a cleaned textual corpus using GROBID.")
    parser.add_argument('--pdf_dir', default=DEFAULT_INPUT_PDF_DIR, help=f"Directory path to the returned PDF articles (default: {DEFAULT_INPUT_PDF_DIR})")
    parser.add_argument('--xml_dir', default=DEFAULT_XML_DIR, help=f"Directory path to store XML/TEI files extracted from PDF articles (default: {DEFAULT_XML_DIR})")
    parser.add_argument('--txt_dir', default=DEFAULT_TXT_DIR, help=f"Directory path to store extracted text files from XML/TEI files (default: {DEFAULT_TXT_DIR})")
    parser.add_argument('--corpus_file', default=DEFAULT_CORPUS_FILE, help=f"TXT file path to store the merged text files as the final corpus (default: {DEFAULT_CORPUS_FILE})")
    parser.add_argument('--grobid_url', default=DEFAULT_GROBID_URL, help=f"GROBID server URL (default: {DEFAULT_GROBID_URL})")

    args = parser.parse_args()

    print("Step 1: Processing PDFs with GROBID...")
    process_pdf_directory(args.pdf_dir, args.xml_dir, args.grobid_url)

    print("\nStep 2: Cleaning and extracting text from XML/TEI...")
    process_xml_directory(args.xml_dir, args.txt_dir)

    print("\nStep 3: Concatenating cleaned texts into final corpus...")
    concatenate_txt_files(args.txt_dir, args.corpus_file)

if __name__ == "__main__":
    main()
