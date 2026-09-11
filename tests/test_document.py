from processing.document_loader import DocumentLoader

loader = DocumentLoader()

document = loader.load("data/uploads/test.pdf")

markdown = loader.export_markdown(document)

print(markdown[:5000])