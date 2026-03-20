# Phase 1 - A Simple Locally Host RAG Workflow

## Overview

The primary goal of this phase of the project is to build out a working RAG system that runs locally on a Macbook Air M3 with 16GB of memory. You should be able to run this with lesser hardware, but you might need to off load LLM inference depending on system limitations. To test the system we will be leveraging an Example  Employee Handbook that I was able to find on the public internet. Along the way we will dive deeper into each part of the system to attempt to better understand how each part functions and relates to the others.

The goal of this phase is to come out the other side with an understanding of why each part of a RAG system exists and an idea of how each part functions. We will intentionally be implementing simple solutions to a lot of these problem. That in turn means at times we will intentionally be implementing a less optimal solution in trade off for something that can easily be implemented and understood.

As far as the Vector Store is concerned we will be leveraging NumPy `ndarray`s and using the `numpy.savez()` to save the arrays for persistent storage. As we move into the later phases we will implement more advanced and purpose built Vector Stores.

To recap what we learned in [Phase 0](phase_0.md) lets run through the different parts of a RAG workflow.

**Loader**

The main job of the loader is to take in some source material and convert it into a common data format prior to handing things off to the Chunker. In basically all RAG systems the target data format is just raw text that we will later convert to Vectors.

To keep things simple we will only be leveraging a PDF loader for this phase as our source document, the Example Employee Handbook, is a PDF.

For this locally hosted phase I have decided on leveraging PyMuPDF as our PDF Loader. It is easy to install and use. There are many other options for tools that can read and translate PDFs into plain text, if you decide to expand on this example RAG tool you might want to consider testing other tools to convert your source PDFs into plain text as you might find other options function better for your needs.

**Chunker**

The chunker has the goal of taking in a large set of text and breaking it up into smaller 'chunks'. There are different ways to implement a chunker. Today we will build a fixed size chunker, meaning we just break the source text into chunks of X number of characters. This is the least effective chunking strategy but I believe it is the perfect starting point in our learning journey.

Later on we will explore other chunking strategies and look at some open source tools that we can leverage to add these different chunking methods to our RAG workflow.

**Embedder**

The job of the embedder is to take the chunks of text and transform them into Vectors that will later be stored in the Vector Store. A vector is a long list of floating point numbers that represent the _semantic_ meaning of the chunk. The loose idea is that if you have three chunks of text, two are about law and the third one is cars, we would expect that the two chunks about law to result in Vectors that are mathematically close while the chunk about cars would not be mathematically close to the other two vectors.

I know this is a bit confusing but it does work and it is pretty amazing to see in action. As you work through this phase of the project you will have the chance to see this in action in a way that drastically helps you understand how things work.

An embedding model is a neural network that has likely been trained using a Semantic Textual Similarity (STS) dataset. A STS dataset consists of a lot of pairs of sentences which have been given a score that represents how close they are from a _semantic_ view point. For example "I like Cheese" and "Cheese is amazing" would have a high score while "Dog is mans best friend" and "The sky is blue" would have a low score.

Today we will be leveraging _all-MiniLM-L6-v2_ as the Embedder. This model is said to have high performance while maintaining good quality in generic use cases. There are models that are specifically trained on Question / Answer style workflows which we might explore further in the later phases of this project.

[https://www.sbert.net/docs/sentence_transformer/pretrained_models.html](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) is a great source of information on commonly available pre-trained embedding models.

One important thing that needs to be understood is that you MUST use the same embedding model during the ingestion phase as you do during the querying phase. This is because during querying we are looking to find values in our Vector Store that are mathematically close to the users input. If we were to use different embedding models we would be comparing apples to oranges which would result in a failure to find the relevant data in our Vector Store.

**Vector Store**

The job of the Vector Store is to do just that, store our vectors for later retrieval.

Today we are going to leverage NumPy as our Vector Store, most would not qualify NumPy as a real Vector Store as it lacks almost all of the functionality that you would find in purpose built Vector Stores. We will explore other Vector Stores as we move forward through this project. But for this first phase NumPy will provide us with everything we need.

**Retriever**

The Retrievers job is to find vectors in the Vector Store that are a close match to the users input. Most of the time your Vector Store will provide its own Retriever. Today we will be building our own Retriever to ensure that we fully understand the role and responsibility of the Retriever.

**LLM**

The role of the LLM is to generate a response to the Users Input.

The LLM will be provided with the Users Input, the related chunks of text as identified by the Retriever, and additional directions that we will provide to help guide the LLM in generating an appropriate response to the user.

To run a LLM model you have a few options. I have found that `LM Studio` is the most user friendly option when it comes to running LLMs locally on Mac based systems. `LM Studio` also supports Windows and Linux, but I have not tried it on these operating systems so be advised your mileage may very.

As far as the specific Large Language Models, today we will focus on the `lfm2.5-1.2b` model by Liquid AI and a quantized version of Qwen3.5 9b are both small enough to run without issue on a 16GB Macbook Air M3 without issue. I have found that the Liquid AI model is fast, light weight, and provides decent results when used in a RAG workflow. The Qwen3.5 9B model is a 'thinking' model which in turn can produce higher quality results with less hallucinations but at the cost of much longer inference times as the model will effectively discuss and modify its answer with itself prior to providing a response.

## Environment Setup

I am going to run with the assumption that the reader has a working understanding of Python Virtual Environments and basic BASH like commands.

If you run into any issues I recommend leaning on AI to help you troubleshoot, if you haven't already tried using something like Claude or Gemini to help you fix an issue with your computer, I believe you will be surprised by how good it is. Although I have a feeling that if you are reading this you have already spent a fair amount of time playing with LLMs for all kinds of different use cases.

Please note that I am working from a Macbook running Tahoe 26.3, so you might need to modify some of these commands, but I do expect them to work without issue on most operating systems that have a working Python3 install.

**Step 1) Create a working directory**
```bash
mkdir RAGity_Ann_Local
cd RAGity_Ann_Local
```

**Step 2) Create Virtual Environment**
```bash
python3 -m venv .venv --prompt RAGity_Ann_Local
```

**Step 3) Activate Virtual Environment**
```bash
source .venv/bin/activate
```

**Step 4) Install Python Dependencies**
```bash
pip install pymupdf sentence-transformers numpy openai
```

* `pymypdf` - Used to convert our PDF into plain text.
* `sentence-transformers` - Used to load and run our Embedding Model to create vectors.
* `numpy` - Used to store our Vectors and provide mathematical functions for vector comparison.
* `openai` - Used as an easy way to interface with the LLM running in LM Studio.

**Step 5) Testing Python Install

Run the following command for your Terminal to open an Interactive Python Shell.

```bash
python3
```

You should see `>>>`. You are now inside of an interactive python shell and can now run little bits of Python code from here. Use the shell to play with the examples as provided below.

Once you are inside of your Interactive Python Shell type in the following lines.

```python
import pymupdf
import numpy
from sentence_transformers import SentenceTransformer
```

Each of the import statements should complete without any errors.

If you are running into errors ensure that you are in the same directory that you created your Python Virtual Environment and also that you have activated your Virtual Environment.

**Step 6) Install LM Studio
[https://lmstudio.ai/download](https://lmstudio.ai/download)


At this point you should be ready start building your own simple RAG workflow that runs locally on your laptop.

## Building the Ingestion Pipeline

The basic ingestion pipeline is as follows

Loader -> Chunker -> Embedder -> Vector Store

In this section we will start to build out our first of two scripts in this RAG workflow. By the end of this section you will have a file that contains both chunks of text and their associated vectors that we can will leverage later.

A link to the completed script will be provided at the end of the section.

What I recommend for now is to work through this from the Interactive Python Shell.
If you haven't already open two terminals, one will be used for the Python Shell and the other will be used for any command line steps that we require.

### Loader

As discussed above a loader has the job of converting your source document into plain text that can later be fed into the Chunker.

Today we will be leveraging the following PDF as our source data.

[Example Employee Handbook](https://publiccounsel.org/wp-content/uploads/2021/12/Employee-Handbook-for-Nonprofits-and-Small-Businesses.pdf)

You can run the following command from your terminal to download the Example Employee Handbook.

```bash
curl -O https://publiccounsel.org/wp-content/uploads/2021/12/Employee-Handbook-for-Nonprofits-and-Small-Businesses.pdf
```

> Note: the `-O` (uppercase o) tells `curl` to download the provided file and save with the same filename.


For our PDF loader we are going to leverage [PyMuPDF](https://github.com/pymupdf/PyMuPDF).

Lets start by importing the PyMuPDF library and using it to read in the PDF.

```python
import pymupdf

doc = pymupdf.open('Employee-Handbook-for-Nonprofits-and-Small-Businesses.pdf')
```

When I am playing with a Python library for the first time I like to leverage `help()` and `dir()`. I find these two steps give me a very good high overview of what I can do with the resulting object.

**dir()**

```python
# List methods of an Object
dir(doc)

# pprint can help for readability
from pprint import pprint
pprint(dir(doc))
```

**help()**

```python
# Detailed help for an Object or Library

# View the help file for the whole library
help(pymupdf)

# Help file for the object we made of the PDF.
help(doc)
```

The thing about PDFs is that they organized in pages.

So if we want to view the content with our loaded PDF object called `doc` we will need to select a specific page to see the associated content.

When looking at the output of `dir(doc)` we see a method that can be helpful to us called `get_page_text`.

Lets leverage the `help()` function to better understand what this method does.

```python
help(doc.get_page_text)
```

We can see that this is a helper function that takes in a page number as the first argument.

```python
doc.get_page_text(1)
```

Running the `get_page_text` function with a page number returns the page content. This is a great starting point but the document has multiple pages. Right now our goal is to get all of the text from the document so that we can pass it to the Chunker.

When looking at `help(doc)` we see two things that might be of interest to us `page_count` and `pages`.

```python
doc.page_count
```

The above code will return the number of pages in the document. We could use this to loop over each page in a for loop.

```python
for page in range(doc.page_count):
    print(doc.get_page_text(page))
```

The above code prints the content of all of the pages in the PDF as plain text. This is very close to what we are looking for.

The next step in our workflow is to feed the Chunker with the complete text of the PDF.

The above code is great for printing the pages of the PDF all at once but it doesn't give us anything we can pass into the Chunker. We still need to add the content into a variable. Lets write some Python code to do just that.

```python
pdf_content = ''
for page in range(doc.page_count):
    pdf_content += doc.get_page_text(page)
```

Above we created an empty string variable called `pdf_content` and then we appended the text from each page to that variable.

If we print out that variable we can see that it contains the full contents of our PDF in text form.

```python
print(pdf_content)
```

I find it best to wrap code like this into functions so we can leverage them later on without needing to type out all of the code again.

You don't need to type out the below code, but I am adding it here for visibility. The code below will be included in the complete script I link to at the end of this section.

```python
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
```

When we need to call this function we will do so like this. If you followed along with the above sections you don't need to run the code below as we already accomplished this step.

```python
pdf_file = 'Employee-Handbook-for-Nonprofits-and-Small-Businesses.pdf'

pdf_content = pdf_to_text(pdf_file)
```

We have a variable called `pdf_content` that contains the content of our PDF.
Now we can move onto the Chunker phase of this project.

### Chunker

The job of a Chunker is to split the content up into small sections of text that will be later converted into Vectors by the embedding model.

The real goal is to ideally break text up into sections that represent a single topic or semantic meaning. For example if we have a paragraph that is all about what to do when you get injured at work, that could probably be its own chunk. But if you have a paragraph that combines multiple concepts, it would be best to break the paragraph up into separate chunks that represent a single concept.

Here we are going to take the most simple route and just break the text up into fixed sizes. This is easy to accomplish in code and gives us a good starting point that we can improve on as we work to refine the RAG system.

To accomplish our fixed size chunking we are going to leverage the String Slicing feature of Python.

```python
pdf_content[0:10]
```

The above code will provide us with the first 10 letters of the string `pdf_content`.

In Python when you are slicing a string you provide the index value of the first character you want to extract and then the index value just after the last character you want to extract. This is called inclusive and exclusive. Starting index value is inclusive and the end index value is exclusive.

For example if we had two strings.

```python
str1 = "0123456789"
str2 = "ABCDEFGHIJ"
```

And we did the following

```python
print(str1[1:5])
print(str2[1:5])
```

We would get back.

```python
1234
BCDE
```

This is because we are selecting index 1 through index 4, index 4 because the second number in a python slice is exclusive.

Now that we understand string slicing lets write a simple function that leverages string slicing to split text into chunks.

```python
def simple_chunker(text, chunk_size=1000):
    """
    Splits text into chunks of 'chunk_size' characters.
    Default 'chunk_size' of 1000.
    """
    chunks = []

    # Iterate through the text in steps of chunk_size
    for offset in range(0, len(text), chunk_size):
        chunk = text[offset : offset + chunk_size]
        chunks.append(chunk)

    return chunks

chunks = simple_chunker(pdf_content, chunk_size=1000)
print(f"Created {len(chunks)} chunks.")
```

Above we have a function that takes in a block of text and splits it up into chunks.

What we are doing is leveraging the `range()` function to help us loop over our text.
This function works like this `range(start, stop, step)`.

So we are saying we want a range of numbers where the first value is 0 in our case, the last value is whatever the length of our string is, and we want to each value returned to be a chunk size apart.

For example

```python
for i in range(0, 50, 10):
    print(i)
```

Would output the following numbers

0, 10, 20, 30, 40

This is because we are stepping by 10, but we don't see 50, because like in string slicing the stop value is exclusive.

At the end of the day all you need to understand is that the end result is a list that contains multiple strings that contain our different chunks of text from the PDF.

This is a very simple chunker. Later on we will explore other chunking strategies which do lead to better accuracy in the responses from our RAG system. But for now we will stick with this fixed size implementation.

### Embedder

Next up we are going to setup the embedding model to take all of our chunks of text and turn them into Vectors based on the _semantic_ meaning of each chunk.

For this step in the process we will be leveraging the [SBERT Sentence Transformers](https://sbert.net) library. The SBERT Sentence Transformers library supports many [pre-trained embedding models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) out of the box. I have decided on using a general embedding model called [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

If you would like to learn a little bit more about the `all-MiniLM-L6-v2` model you can learn more at this [Huggingface discussion page](https://discuss.huggingface.co/t/train-the-best-sentence-embedding-model-ever-with-1b-training-pairs/7354)

Lets start playing around with the SBERT Sentence Transformer library to better understand how it works.

```python
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
```

Please note that importing the SentenceTransformer library can take a bit of time to complete, so if running from the Interactive Python Shell expect to wait for the import command to hang for a little bit before presenting you with a '>>>' again.

A second thing to take into consideration is that when you load the embedding model it need to be downloaded to your machine which might also take a little time. The embedding model we are leveraging today is about 80MB in size.

Lets play around with the embedding model.

```python
test_sentences = ['The quick brown fox jumps over the lazy dog', 'Pack my box with five dozen liquor jugs']

# Encode the sentences as vectors
vectors = embedding_model.encode(test_sentences)
```

What we end up with is a multidimensional array called `vectors` that contains a list of numbers for each sentence/chunk provided. These numbers represent the _semantic_ meaning of the sentence/chunk.

```python
# Show the number of 'vectors' in the Array
len(vectors)

# View what a single vector looks like
print(vectors[0])
```

Hopefully now we have a pretty good understanding of what a vector is.

Below is everything we need to turn a chunk of text into a Vector using our embedding model of choice.

```python
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def chunks_to_vector(chunks):
    """
    Takes in a list of text chunks.
    Converts them to vectors via an embedding model.
    Returns an ndarray of vectors.
    """
    print(f"Converting {len(chunks)} chunks into vectors.")
    vectors = embedding_model.encode(chunks)

    return vectors

vectors = chunks_to_vector(chunks)
```

### Vector Store

The main goal of a Vector Store is to keep data for later retrieval. We could skip saving the data and just run through the initial steps every time we launched our script, but that wouldn't be very efficient. To prevent us from needing to reprocess our source document every time we restart the program we need to save our chunks of text and their vectors to a file.

NumPy has a `.save()` method which allows us to save an array to disk. This is great but we have two arrays that we need to save. Just saving the Vector values isn't good enough because we can't go from the Vector back to the original text. We must also keep a copy of our original chunks of text.

We could save both arrays in their own individual files but a better solution is to leverage the `numpy.savez()` function. This feature allows for storing multiple arrays in a single file.


```python
import numpy

def save_vector_store(filename, chunks, vectors):
    """
    Take in our Chunks, and Vectors and save them as a .npz for long term storage
    """
    # Convert the chunks list to a Numpy Array
    chunks_array = numpy.array(chunks)

    # We should have the same number of chunks as vectors
    if chunks_array.shape[0] != vectors.shape[0]:
        raise ValueError(f"Chunks Array size {chunks_array.shape[0]} not equal to Vector Array Size {vectors.shape[0]}")

    print(f"Saving Chunks and Vectors to {filename}")
    numpy.savez(filename, chunks=chunks_array, vectors=vectors)

save_vector_store("phase_1_vector_store.npz", chunks, vectors)
```

The above code saves our vectors and chunks to a single file for later use. This function converts our python `list` of chunks into a numpy array, checks to ensure that the number of chunks matches the number of vectors, and then passes them both to the `numpy.savez()` function.

If you have followed along with the above steps you should now have a .npz file saved which contains both our Chunks and also our Vectors.

### Ingestion Workflow Summary

At this time we have the following figured out.

Loader -> Chunker -> Embedder -> Vector Store

I have included a file called [phase_1_ingest.py](phase_1_ingest.py) which is a complete script that represents everything we have covered above.

Next we will move onto the Querying part of this simple RAG workflow.

## Building the Query Pipeline

Now we are going to start working on building out the Query Pipeline.

User Input -> Embedder -> Retriever -> LLM -> Respond to User

If you are starting from here you can execute the [phase_1_ingest.py](phase_1_ingest.py) script to create and save the Vector Store. Once you have ensured that you have a copy of the vector store saved to disk launch a new Python Interactive Shell and follow along.

```bash
python3 phase_1_ingest.py
```

In this section we will build out a system that will take in a user question about the employee handbook, locate associated text in the handbook, pass the associated text along with the user question to the LLM and provide the LLMs response back to the user.

This is where the real power of RAG workflows come into play. We are only passing relevant information to the LLM helping keep our Context Window clean, small, and exact. This helps us ensure that we doing our best to setup the LLM for success.

Now lets start a new Python Interactive Shell and start working on building out the next part of our RAG workflow.

```bash
python3
```

### Loading the Vector Store into Memory

The first thing we need to do is load our Vector Store (phase_1_vector_store.npz) into memory.

```python
import numpy

vector_store = numpy.load("phase_1_vector_store.npz")

chunks = vector_store["chunks"]
vectors = vector_store["vectors"]

print(vector_store)
print(chunks)
print(vectors)
```

This loads our Vector Store into memory. One thing that we should be aware of is when using this method the size of our Vector Store will impact our memory usage. This is probably going to be one of the early limitations on how large of a dataset we can vectorize and search using NumPy as our Vector Store. Using more purpose built RAG tools we can work around this linear relation of size to memory usage by Indexing the data, this allows us to load the Index into memory while keeping the majority of the data at rest on disk.

### Embedding the User's Question

The next step in our RAG workflow is collecting the user's question about the employee handbook.

For simplicity we are going to leverage the `input()` function to ask the user for their question about the Employee Handbook.

```python
user_question = input("What is your question?: ")
```

For this example lets enter the following question.
"Am I allowed to take time off to vote?"

```python
print(user_question)
```

As we have discussed in the past, for us to find relevant data in the Vector Store we need to leverage the Embedding Model again to convert the users question into a Vector representation of the _semantic_ value of the user input.

We will need import the Embedding Model just like we did in the ingest workflow above. If you made any changes around the Embedding Model in the ingest workflow they will need to be replicated here.

```python
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

user_question_vector = embedding_model.encode(user_question)

print(user_question_vector)
```

The code above loads the `all-MiniML-L6-v2` model into memory, passes the model the user's question for encoding, and then stores the resulting vector into `user_question_vector`. We will use this new vector to search for similar vectors in our Vector Store.

Again it is very important that we leverage the same Embedding Model along with any additional settings that were leveraged during the ingestion workflow.

### Retriever / Cosine Similarity

The Retriever step takes the vector representation of the user's question and compares it against the vectors in our Vector Store. There are a few methods that we can implement to locate similar Vectors. Today we will be using the Cosine Similarity method as it is easy to code and hopefully easy to understand. Please note that our implementation is not optimized for performance, I have decided to go with a more explicit method to help you better understand the process.

The formula for Cosine Similarity is `dot(A, B) / (norm(A) * norm(B))` where A is the user question vector and B is a Vector we would like to compare similarity against.

The result of Cosine Similarity is a value between -1 and 1. 1 translates to identical meaning, 0 is unrelated, and -1 denotes opposite meaning.

If we compared vectors representing "Cat" and "Cat" we would expect a resulting value of 1. If we compare to vectors that are unrelated like "Dog" and "Star" we would expect a value around 0. If we compare to opposing vectors like "Good" vs "Bad" we would expect a value close to -1.

To locate vectors in the Vector Store that are relevant to the user's input we need to compare the user's input vector against each vector in the Vector Store and log the 'similarity' value.

```python
import math

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
```

Function that computes our similarity between two vectors using the Cosine Similarity method. We will not be diving into Dot Products or Normalization in this guide. But if you are interested in learning more I recommend pulling up your favorite LLM and asking it about how Cosine Similarity works.

```python
similarities = []

for i in range(len(vectors)):
    score = cosine_similarity(user_question_vector, vectors[i])
    similarities.append(score)
```

Now we need to compare the similarity of our user question vector against each vector in the Vector Store. This loop we are doing is not optimal at all and there are better ways to do this leveraging some NumPy functions. But in the interest of understanding the process as a whole I have decided to pick the less efficient method that is more simple to understand.

The above code will build a list that contains the Similarity value between the users input and each chunk vector in the Vector Store.

```python
print(len(similarities))
print(similarities)
```

We can see that the similarities list is the same size as our original number of chunks that we created when ingesting the Employee Handbook PDF. This is to be expected as we expect one vector stored per chunk ingested.

When we look at the values stored in the similarities list we see that they are floating point numbers ranging from -1 to 1. As a reminder 1 is a perfect match, 0 is unrelated, and -1 is an opposite semantic meaning.

These similarity numbers are great in all but what we really need right now is to find what vectors have a high similarity value as compared to the user's input.

```python
# Pair each score with its index, sort by score descending
ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

print(ranked)
```

The code above gives us a list that is shaped like this `[(index_number, similarity_score), (index_number, similarity_score), (index_number, similarity_score)...]`.

We have a sorted list that includes the index of each vector along with its similarity score. We could provide the LLM with each chunk of text along with the similarity score, but that isn't very helpful. We want to only provide the LLM with information that is relevant to the user's input.

One solution to the problem of providing the LLM with only relevant information is to leverage the Top K method. In basic it just means we provide the top X number of matches. It is simple to implement and in a lot of situations it can get the job done. It is also easy to understand and implement which is why we are going to use it today.

```python
results = []
for i, score in ranked[:5]:
    results.append(chunks[i])
    print(f"Score: {score: .4f}")
    print(f"Chunk: \n{chunks[i]}")
    print(f"--------------------------------------------------------------")
    print(f"--------------------------------------------------------------")
    print()
```

The code above we show us the Similarity Score and the associated chunk of the top 5 results. This works because we have sorted the list before slicing the array named `ranked`. It is important that we sort prior to implementing Top K. If we didn't sort we would be returning the first X items in our list which would probably relate to just the first chunks ingested into the Vector Store.

On my machine the first result has a `Score: 0.6341` which is roughly 63% similarity.

Reviewing the chunk I can see that it holds most of the section "M. Time Off To Vote" but this section has been cut in half. The reason it has been cut in half is because of our suboptimal chunking strategy. If you recall we decided on the simple method of breaking the Employee Handbook into fixed size chunks.

The second chunk returned has a `Score: 0.4159` and it contains the second half of the associated section in the Employee Handbook.

The remaining three chunks have a score between 0.2696 and 0.2847
Looking at the values in the text we can see that they relate to topics like time off and legal. But these chunks are not directly related to the users question. This is a great example of why just leveraging Top K when selecting relevant Vectors isn't exactly perfect. We will move forward with Top K but think about what are some better ways that we could identify what Chunks to provide to our LLM.

Before we move on lets put everything into a nice block of code so we can get the full picture while also allowing for easy usage later.

```python
import math

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

results = retriever(user_question_vector, vectors, chunks, top_n=5)
```

### Prompt Construction

We are officially moving on to interacting with the LLM.

The first thing we need to do is construct the prompt that we are going to send to the LLM.

Our goal is to provide the LLM with enough detail to help it accomplish the task at hand. In our workflow the task is to answer a user's question about the Employee Handbook.

What we need to do is provide the LLM with the following.
 * Top chunks as gathered from the Retriever step
 * The user's original question
 * Instructions that tell the LLM how to behave.

```python
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

print(prompt)
```

Above we are creating `chunks_of_text` using the `.join()` function. This might be a little confusing at first glance. The `.join()` method is a pythonic way of taking a list of converting it into a string that contains all of the values inside of the list. It is nothing more than a quick way to write a loop that appends the values to a string, in this case adding two new lines between each value.

After that we construct our prompt. We give the LLM some context on what we are asking it to do along with providing it our most relevant entries in the Vector Store and also the user's question.

### Calling the LLM via LM Studio

We are almost across the finish line. What we are going to do next is setup an LLM to run locally. Sadly this is the most 'black box' part of the process and also might cause you the most issues. If you get stuck during this process I recommend taking the complete prompt as created above and just pasting it into our favorite LLMs chat interface and seeing what the result is.

I am using a Macbook Air M3 with 16GB of memory. If you are using something different I still think it is worth attempting to follow along.

We will be leveraging LM Studio today as I feel it is the most simple option to get an LLM running locally.

[LM Studio Download Page](https://lmstudio.ai/download)

Running LLMs locally could be a whole set of posts. With that I won't be talking about how to get things running with LM Studio today, I recommend watching a quick YouTube video to help you get started. In basic you have a few buttons on the left hand side of the UI. Click around until you find two things. Where to download Models and how to run the server on TCP/IP port `:1234`.

To get started I would like to start with the smallest 'modern' MLX based model that I am aware of. It is `liquid/lfm2.5-1.2b` this clocks in at about 1.25GB which should fit nicely on any modern Macbook.

There are many great tutorials available online around how to use LM Studio so I will just be providing broad strokes.

1. Download `liquid/lfm2.5-1.2b`
2. Start LM Studio Server
3. Load Model
4. Test Connection from Python to LM Studio

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

response = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "user", "content": "You are a helpful assistant. Say 'Connection Successful!'"}
    ],
)

print(response.choices[0].message.content)
```

The above code will create a client that leverages the OpenAI API to interact with LM Studio that should be listening on port 1234. With LM Studio you can have a generic value in the `model` field, LM Studio will always just use the model that you loaded from the Developer / LM Studio Server page.

We then send a message to the LLM for processing.

Now that we have a working interface into our LLM lets wrap things up into a simple function to make this process a little more simple.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def message_llm(user_prompt):
    response = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "user", "content": user_prompt}
        ],
    )

    return response.choices[0].message.content

response = message_llm(prompt)

print(response)
```

Assuming everything is working as expected you should received an answer to the user's question from the LLM! During my testing I get the following result when leveraging `liquid/lfm2.5-1.2b` and asking it `Am I allowed to take time off to vote?`.

```markdown
Response From: liquid/lfm2.5-1.2b

Yes, you are allowed to take time off to vote. According to the policy, [ORGANIZATION] will grant up to 2 hours of paid time off for voting if your schedule prohibits you from voting during nonworking hours. You should request this time off from your supervisor at least two working days prior to the election.
```

That is it we now have a fully functional RAG workflow that runs locally.

From here there are a lot of optimizations that could be made to the workflow. Later on we might revisit this local workflow and implement these optimizations. But at the current time the plan is to move onto the Self Hosted workflow where will start to leverage more purpose built tools that provide out of the box solutions to some of the problems we ignored in this phase of this guide on building your own RAG workflows.

### Querying Workflow Summary

At this point we have built a very simple RAG system leveraging very few tools.

Loader -> Chunker -> Embedder -> Vector Store -> Retriever -> LLM

I hope this has helped demystify how a RAG system works under the hood and shows you that it is possible to run a RAG workflow totally locally without too much effort.

Obviously this workflow has a lot of room for improvement.

## Putting It All Together

I have taken all of the code we covered in this document and wrapped them up into two files.

[phase_1_ingest.py](phase_1_ingest.py)
[phase_1_query.py](phase_1_query.py)

To run these scripts you will need to have LM Studio running with model loaded and the server enabled on port `:1234`. You will need a PDF that you would like ingest into the Vector Store. You will also need to have these Python packages installed.

```bash
pip install pymupdf langchain-text-splitters sentence-transformers numpy openai
```

I hope that this document was helpful in learning about how RAG workflows work and please stay tuned for the next phase of this project where we will be building out a RAG workflow that runs on cloud infrastructure and leverages more purpose built tools.
