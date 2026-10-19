"""Check report completeness, output alias and document text/rule colours."""
import hashlib
from pathlib import Path
from pypdf import PdfReader
from pypdf.generic import ContentStream
ROOT=Path(__file__).resolve().parents[1]

def main():
    file=ROOT/'output'/'230145E_a03.pdf'
    reader=PdfReader(file)
    text='\n'.join(page.extract_text() for page in reader.pages)
    for required in ['Dilhan W.A.','230145E','en3160-assignment-03','sigmoid','LeNet','ResNet18',
                     'cross-entropy','validation','Reproducibility']:
        assert required.lower() in text.lower(),f'Missing {required}'
    assert 'Experimental protocol' in reader.pages[0].extract_text(),'First page must contain report content'
    assert 4<=len(reader.pages)<=18
    for i,page in enumerate(reader.pages):
        content=ContentStream(page.get_contents(),reader)
        for operands,operator in content.operations:
            if operator in [b'rg',b'RG']:
                assert all(abs(float(x))<1e-9 for x in operands),f'Coloured document drawing on page {i+1}: {operands}'
    assert hashlib.sha256(file.read_bytes()).digest()==hashlib.sha256((ROOT/'output'/'230145E_a02.pdf').read_bytes()).digest()
    print(f'Verified {len(reader.pages)} pages, identity, repository, required sections, black vector colours, and identical submission-name alias.')

if __name__=='__main__':main()
