import fitz  # PyMuPDF
import pdfplumber
from typing import Dict, Any, List, Optional
import io
import os


class PDFProcessor:
    def __init__(self):
        self.supported_formats = ['.pdf']

    async def extract_text(self, file_path: str) -> Dict[str, Any]:
        """Extract text from PDF file"""
        try:
            # Method 1: PyMuPDF for speed
            text_pymupdf = self._extract_with_pymupdf(file_path)

            # Method 2: pdfplumber for accuracy with tables
            text_pdfplumber = self._extract_with_pdfplumber(file_path)

            # Combine results
            combined_text = text_pymupdf + "\n\n" + text_pdfplumber

            return {
                "success": True,
                "text": combined_text[:10000],  # Limit text length
                "char_count": len(combined_text),
                "methods_used": ["PyMuPDF", "pdfplumber"]
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }

    def _extract_with_pymupdf(self, file_path: str) -> str:
        """Extract text using PyMuPDF"""
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            print(f"PyMuPDF extraction error: {e}")
        return text

    def _extract_with_pdfplumber(self, file_path: str) -> str:
        """Extract text using pdfplumber"""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

                    # Try to extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            # Convert table to readable text
                            table_text = self._table_to_text(table)
                            text += table_text + "\n"
        except Exception as e:
            print(f"pdfplumber extraction error: {e}")
        return text

    def _table_to_text(self, table: List[List]) -> str:
        """Convert table data to readable text"""
        if not table:
            return ""

        text_lines = []
        for row in table:
            # Filter out None values and join with tabs
            row_text = "\t".join([str(cell) if cell is not None else "" for cell in row])
            text_lines.append(row_text)

        return "\n".join(text_lines)

    async def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract PDF metadata"""
        try:
            doc = fitz.open(file_path)
            metadata = doc.metadata

            # Additional metadata
            metadata["page_count"] = len(doc)
            metadata["file_size"] = os.path.getsize(file_path)

            # Estimate word count from first few pages
            sample_text = ""
            for i in range(min(3, len(doc))):
                sample_text += doc[i].get_text()
            metadata["estimated_word_count"] = len(sample_text.split())

            doc.close()

            return {
                "success": True,
                "metadata": metadata
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "metadata": {}
            }

    async def search_in_pdf(self, file_path: str, search_terms: List[str]) -> Dict[str, Any]:
        """Search for terms in PDF"""
        try:
            doc = fitz.open(file_path)
            results = {}

            for term in search_terms:
                term_results = []
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text_instances = page.search_for(term)

                    if text_instances:
                        # Get context around each occurrence
                        page_text = page.get_text()
                        for inst in text_instances[:3]:  # Limit to first 3 per page
                            # Simple context extraction
                            term_start = page_text.find(term)
                            if term_start != -1:
                                context_start = max(0, term_start - 100)
                                context_end = min(len(page_text), term_start + len(term) + 100)
                                context = page_text[context_start:context_end]

                                term_results.append({
                                    "page": page_num + 1,
                                    "context": context,
                                    "count": len(text_instances)
                                })

                if term_results:
                    results[term] = term_results

            doc.close()

            return {
                "success": True,
                "results": results,
                "total_terms_found": sum(len(v) for v in results.values())
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": {}
            }

    async def extract_images(self, file_path: str, output_dir: str) -> Dict[str, Any]:
        """Extract images from PDF"""
        try:
            doc = fitz.open(file_path)
            images = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()

                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)

                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # Save image
                        image_filename = f"page_{page_num + 1}_img_{img_index}.{image_ext}"
                        image_path = os.path.join(output_dir, image_filename)

                        with open(image_path, "wb") as f:
                            f.write(image_bytes)

                        images.append({
                            "page": page_num + 1,
                            "index": img_index,
                            "path": image_path,
                            "size": len(image_bytes),
                            "format": image_ext
                        })

            doc.close()

            return {
                "success": True,
                "images": images,
                "total_images": len(images)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "images": []
            }

    async def convert_to_markdown(self, file_path: str) -> Dict[str, Any]:
        """Convert PDF to markdown format"""
        try:
            # Extract text
            text_result = await self.extract_text(file_path)

            if not text_result["success"]:
                return text_result

            text = text_result["text"]

            # Simple conversion to markdown
            lines = text.split('\n')
            md_lines = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Convert headings (simple heuristic)
                if line.isupper() and len(line) < 100:
                    md_lines.append(f"## {line}")
                elif line.endswith(':') and len(line) < 50:
                    md_lines.append(f"### {line}")
                else:
                    md_lines.append(line)

            markdown = '\n\n'.join(md_lines)

            return {
                "success": True,
                "markdown": markdown,
                "original_length": len(text),
                "markdown_length": len(markdown)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "markdown": ""
            }