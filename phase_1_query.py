import numpy
import math
from sentence_transformers import SentenceTransformer
from openai import OpenAI

vector_store = numpy.load("phase_1_vector_store.npz")
chunks = vector_store["chunks"]
vectors = vector_store["vectors"]

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def cosine_similarity(vector_a, vector_b):
    dot_product = 0
    norm_a = 0
    norm_b = 0

    for i in range(len(vector_a)):
        dot_product += vector_a[i] * vector_b[i]
        norm_a += vector_a[i] ** 2
        norm_b += vector_b[i] ** 2

    norm_a = math.sqrt(norm_a)
    norm_b = math.sqrt(norm_b)

    return dot_product / (norm_a * norm_b)

def retriever(user_question_vector, vectors, chunks, top_n=5):
    similarities = []

    for i in range(len(vectors)):
        score = cosine_similarity(user_question_vector, vectors[i])
        similarities.append(score)

    # Pair each score with its index, sort by score descending
    ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    results = []

    for i, score in ranked[:5]:
        results.append(chunks[i])

    return results

def message_llm(user_prompt):
    response = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "user", "content": user_prompt}
        ],
    )

    return response.choices[0].message.content



user_question = input("What is your question?: ")
user_question_vector = embedding_model.encode(user_question)
results = retriever(user_question_vector, vectors, chunks, top_n=5)

chunks_of_text = "\n\n".join(results)

prompt = f"""You are the LLM in a RAG workflow.

Use are going to be provided with a users question and multiple chunks of text that should contain the required information to answer the users question.

If the answer is not found in the chunks of text just respond with I don't know.

<chunks_of_text>
{chunks_of_text}
</chunks_of_text>

<user_question>
{user_question}
</user_question>
"""

response = message_llm(prompt)
print(response)

exit()
