"""
PDF Ingestion Module

This module handles the extraction of text content from PDF files.
It uses the pypdf library to read PDFs and extract text page by page.
"""

from typing import List, Dict
from pypdf import PdfReader


def extract_pdf_pages(file_path: str) -> List[Dict]:
    """
    Extract text content from each page of a PDF file.

    Args:
        file_path: Path to the PDF file to extract text from.

    Returns:
        A list of dictionaries, each containing:
            - text: The string content of the page
            - page_number: The 1-indexed page number

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        Exception: For other PDF reading errors.
    """
    pages_data = []

    try:
        reader = PdfReader(file_path)

        for idx, page in enumerate(reader.pages):
            page_number = idx + 1  # 1-indexed page number

            try:
                text = page.extract_text()

                # Skip pages with None or empty text
                if text is None or text.strip() == "":
                    print(f"Skipping page {page_number}: No extractable text found.")
                    continue

                pages_data.append({
                    "text": text.strip(),
                    "page_number": page_number
                })

            except Exception as e:
                print(f"Error extracting text from page {page_number}: {e}")
                continue

    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading PDF file '{file_path}': {e}")

    return pages_data


def get_page_count(file_path: str) -> int:
    """
    Get the total number of pages in a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Total number of pages in the PDF.
    """
    try:
        reader = PdfReader(file_path)
        return len(reader.pages)
    except Exception as e:
        raise Exception(f"Error getting page count for '{file_path}': {e}")
