# scripts/pdf_utils.py

import requests
import pdfplumber
import io
from loguru import logger


def download_pdf_and_extract_text(pdf_url: str) -> str | None:
    """
    Downloads a PDF from a URL and extracts its text content.

    Args:
        pdf_url (str): The URL of the PDF file.

    Returns:
        str or None: The extracted text or None if an error occurs.
    """
    try:
        logger.info(f"Downloading PDF from: {pdf_url}")
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        if 'application/pdf' not in response.headers.get('Content-Type', '').lower():
            logger.error(f"Invalid content-type: {response.headers.get('Content-Type')}")
            return None

        pdf_file = io.BytesIO(response.content)

        extracted_text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for i, page in enumerate(pdf.pages):
                if i % 10 == 0:
                    logger.debug(f"Extracting page {i + 1} of {len(pdf.pages)}")
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"

        if not extracted_text.strip():
            logger.warning("No text extracted – PDF may be image-based.")
            return None

        logger.info("PDF text extraction completed successfully.")
        return extracted_text

    except requests.RequestException as e:
        logger.error(f"Download failed: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error during PDF processing: {e}")
        return None

