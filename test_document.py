from processing.document_loader import DocumentLoader
from processing.chunker import DocumentChunker


loader = DocumentLoader()

document = loader.load(
    "data/uploads/test.pdf"
)

chunker = DocumentChunker()

chunks = chunker.chunk(document)

print("Total chunks:", len(chunks))

for chunk in chunks[:5]:

    print("\n========================")
    print("CHUNK:", chunk["chunk_id"])
    print("========================")

    print(chunk["text"][:1000])