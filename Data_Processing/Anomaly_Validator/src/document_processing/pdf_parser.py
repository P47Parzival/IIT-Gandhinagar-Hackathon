"""
PDF Parser for extracting text and tables from PDF documents.

Handles invoices, contracts, financial statements, and other PDF-based supporting documents.
"""

import os
from typing import Dict, Any, List, Optional
import PyPDF2
import pdfplumber


class PDFParser:
    """
    Parse PDF documents to extract text, tables, and metadata.
    
    Uses two libraries for comprehensive extraction:
    - PyPDF2: Fast text extraction
    - pdfplumber: Advanced table extraction
    """
    
    def __init__(self, extract_tables: bool = True):
        """
        Initialize PDF parser.
        
        Args:
            extract_tables: Whether to extract tables (slower but more comprehensive)
        """
        self.extract_tables = extract_tables
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse PDF file and extract content.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with:
                - text: Extracted text content
                - tables: List of extracted tables (if enabled)
                - metadata: PDF metadata
                - num_pages: Number of pages
                - file_name: Original file name
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        result = {
            'text': '',
            'tables': [],
            'metadata': {},
            'num_pages': 0,
            'file_name': os.path.basename(file_path),
            'file_path': file_path
        }
        
        try:
            # Extract text with PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                result['num_pages'] = len(pdf_reader.pages)
                result['metadata'] = self._extract_metadata(pdf_reader)
                
                # Extract text from all pages
                text_parts = []
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"=== Page {page_num + 1} ===\n{page_text}")
                
                result['text'] = "\n\n".join(text_parts)
            
            # Extract tables with pdfplumber (if enabled)
            if self.extract_tables:
                result['tables'] = self._extract_tables(file_path)
            
            print(f"✓ Parsed PDF: {result['file_name']} ({result['num_pages']} pages, "
                  f"{len(result['text'])} chars, {len(result['tables'])} tables)")
            
        except Exception as e:
            print(f"⚠️  Error parsing PDF {file_path}: {e}")
            result['error'] = str(e)
        
        return result
    
    def _extract_metadata(self, pdf_reader: PyPDF2.PdfReader) -> Dict[str, Any]:
        """Extract PDF metadata."""
        metadata = {}
        try:
            if pdf_reader.metadata:
                for key, value in pdf_reader.metadata.items():
                    # Remove leading '/' from keys
                    clean_key = key.lstrip('/')
                    metadata[clean_key] = str(value) if value else None
        except:
            pass
        return metadata
    
    def _extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber."""
        tables = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table_idx, table in enumerate(page_tables):
                        if table:
                            tables.append({
                                'page': page_num + 1,
                                'table_index': table_idx,
                                'data': table,
                                'rows': len(table),
                                'cols': len(table[0]) if table else 0
                            })
        except Exception as e:
            print(f"⚠️  Table extraction error: {e}")
        
        return tables
    
    def extract_text_simple(self, file_path: str) -> str:
        """Quick text extraction without tables (faster)."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_parts = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
                return "\n\n".join(text_parts)
        except Exception as e:
            print(f"⚠️  Error extracting text from {file_path}: {e}")
            return ""
    
    def search_in_pdf(self, file_path: str, search_term: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """
        Search for a term in PDF.
        
        Args:
            file_path: Path to PDF
            search_term: Term to search for
            case_sensitive: Whether search is case-sensitive
            
        Returns:
            List of matches with page numbers and context
        """
        matches = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    search_text = text if case_sensitive else text.lower()
                    search_for = search_term if case_sensitive else search_term.lower()
                    
                    if search_for in search_text:
                        # Find context around match
                        lines = text.split('\n')
                        matching_lines = [line for line in lines if search_for in (line if case_sensitive else line.lower())]
                        
                        matches.append({
                            'page': page_num + 1,
                            'matches': matching_lines
                        })
        
        except Exception as e:
            print(f"⚠️  Search error: {e}")
        
        return matches


if __name__ == "__main__":
    # Test PDF parser
    print("Testing PDF Parser...")
    
    parser = PDFParser(extract_tables=True)
    
    # Create a simple test scenario
    print("\n✓ PDF Parser initialized successfully")
    print("✓ Ready to parse PDF documents")
    
    # Note: Actual testing requires a PDF file
    # Usage example:
    # result = parser.parse('path/to/invoice.pdf')
    # print(result['text'][:500])

