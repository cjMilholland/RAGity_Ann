import pymupdf
import numpy
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
pdf_file = 'Employee-Handbook-for-Nonprofits-and-Small-Businesses.pdf'
chunk_size = 1000

def pdf_to_text(pdf_file_name):
    """
    Reads in an PDF and outputs a string containing PDF content.
    """
    doc = pymupdf.open(pdf_file_name)

    print(f"PDF contains {doc.page_count} pages.")

    pdf_content = ''
    for page in range(doc.page_count):
        pdf_content += doc.get_page_text(page)

    return pdf_content

def simple_chunker(text, chunk_size=500):
    """
    Splits text into chunks of 'chunk_size' characters.
    Default 'chunk_size' of 500.
    """
    chunks = []

    # Iterate through the text in steps of chunk_size
    for offset in range(0, len(text), chunk_size):
        chunk = text[offset:offset + chunk_size]
        chunks.append(chunk)

    print(f"Created {len(chunks)} chunks.")
    return chunks

def chunks_to_vector(chunks):
    """
    Takes in a list of text chunks.
    Converts them to vectors via an embedding model.
    Returns an ndarray of vectors.
    """
    print(f"Converting {len(chunks)} chunks into vectors.")
    vectors = embedding_model.encode(chunks)

    return vectors

def save_vector_store(filename, chunks, vectors):
    """
    Take in our Chunks and Vectors and save them in a .npz
    """
    chunks_array = numpy.array(chunks)

    if chunks_array.shape[0] != vectors.shape[0]:
        raise ValueError(f"Chunks Array Size {chunks_array.shape[0]} is not equal to Vector Array Size {vectors.shape[0]}")

    print(f"Saved Chunks and Vectors to {filename}")
    numpy.savez(filename, chunks=chunks_array, vectors=vectors)


# Convert PDF into Text
pdf_text = pdf_to_text(pdf_file)
# Convert Text into Chunks
chunks = simple_chunker(pdf_text, chunk_size)
# Convert Chunks to Vectors
vectors = chunks_to_vector(chunks)
# Save Chunks and Vectors to .npz file
save_vector_store("phase_1_vector_store.npz", chunks, vectors)
