class RAGError(Exception):
    pass

class PDFLoadError(RAGError):
    pass

class CorruptPDFError(PDFLoadError):
    pass

class EncryptedPDFError(PDFLoadError):
    pass

class PDFTooLargeError(PDFLoadError):
    pass

class EmptyDocumentError(PDFLoadError):
    pass

class ChunkingError(RAGError):
    pass

class EmbeddingError(RAGError):
    pass

class VectorStoreError(RAGError):
    pass

class RetrievalError(RAGError):
    pass
