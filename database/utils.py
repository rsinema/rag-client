import pdfplumber
import os
import argparse
from typing import Optional, List

def pdf2txt(pdf_path: str) -> str:
    """
    Convert a PDF file to text format.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Filename of the generated text file
    """
    # Storage for extracted text
    pages_text: List[str] = []
    
    # Open the PDF file
    with pdfplumber.open(pdf_path) as pdf:
        # Iterate through all pages
        for page in pdf.pages:
            # Extract text from the page
            text = page.extract_text()
            
            # Add to pages if there's actual content
            if text and text.strip():
                pages_text.append(text.strip())
    
    # Join all pages with newlines
    text_content = '\n\n'.join(pages_text)
    
    # Create output filename
    txt_filename = os.path.splitext(os.path.basename(pdf_path))[0] + '.txt'
    
    # Write to file
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write(text_content)
    
    return txt_filename

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Convert PDF files to TXT files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python script.py -f /path/to/pdf/files/file.pdf
        '''
    )
    
    parser.add_argument(
        '-f', '--file',
        help='PDF file to convert'
    )

    # Parse arguments
    args = parser.parse_args()
    
    # Convert files
    if args.file:
        # Convert single file
        if not os.path.isfile(args.file):
            parser.error(f"Input file does not exist: {args.file}")
        text_file = pdf2txt(args.file)
        print(f"Created text file: {text_file}")
    else:
        parser.error("Please provide a file to convert")

if __name__ == "__main__":
    main()