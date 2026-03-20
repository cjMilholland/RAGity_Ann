# Phase 0 - What is RAG and how does it work?

## 1. What is RAG and why does it exist?

In the most simple terms Retrieval Augmented Generation (RAG) provide small chunks of relevant data to a Large Language Model (LLM) so that the LLM can answer a user's question leveraging accurate data which the LLM might not have been trained on.

What problems does RAG solve?

LLMs are pre-trained. Meaning that the LLM has been created by training it with large amount of textual data. This data normally is sourced from the internet and suffers from what is commonly referred to as a 'knowledge cutoff date'. What this results in is a LLM that has a vast array of knowledge but the LLMs knowledge is limited to what it has been shown, if the LLM was trained in January of 2025 with the most currently available data, the LLM will only be able to answer questions current to January of 2025 in relation to the data it was trained on.

During this project we will be focusing on building an Employee Handbook search system. The goal of the system is to allow a user to ask questions in natural language and receive answers based on the content that exists in the current Employee Handbook.

In the real world your Employee Handbook wasn't used as training data of an off the self LLM. This means that any Employee Handbook questions that are asked against an off the shelf LLM won't be answered correctly. For example if a user asks "How many vacation days do I get?" the generic LLM will likely respond with a best guess based on its training data, but the answer will likely be incorrect.

Without using RAG you could attempt to solve this problem a couple of ways.

One possible way is to build a system that includes the complete Employee Handbook into the chat session with the LLM. The main issue with this is that when interacting with an LLM you have the concept of a Context Window. The Context Window is the amount of input that the LLM is able to handle while responding to a request. With that it is possible that the employee handbook is too large to fit into the available Context Window of the LLM. Another possible issue is that as the Context Window reaches its max capacity it is common for LLMs to start to behave in odd and unexpected ways. Accuracy and quality of the responses start to diminish along with the occasional 'corrupted' response. In addition to these issues the larger the input to the LLM the greater the compute needed to generate a response. This increase is compute can translate into additional costs to the user.

These challenges make providing the Employee Handbook as part of the input to the LLM a non-starter for most situations.

Another possible solution to the LLM lacking relevant knowledge is to perform what is called Supervised Fine Tuning (SFT). SFT is a process that allows you to take an existing model and further train the model on a predetermined dataset. In our example situation the training dataset would include the Employee Handbook along with a set of question and answer pairs that help train the LLM on the new information. There are a couple of challenges with SFT, first it is a complicated, slow, and can be costly. The second is that any time you update the Employee Handbook you would need to run through the SFT process again. Third is if you decide to change your model or upgrade to a new version you will need to refine your SFT process to ensure high quality results from the new LLM.

These issues were problematic enough that Retrieval Augmented Generation was created. RAG was first described in a paper titled "[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401)".

The basic idea behind RAG is to first collect only the relevant data to the users query and provide that to the LLM along with the question. This saves us from needing to do SFT while also not filling up the Context Window with unnecessary data. RAG also helps with hallucination as the LLM has been provided with the hard facts that it can directly reference.

## 2. The RAG two step: Ingest and Query

At a high level RAG is comprised of two fundamental steps. Ingestion and Querying.

Lets use the Employee Handbook example to further explain. To ingest the Employee Handbook we would first need to break the Employee Handbook into chunks of text that are roughly N words long with each chunk ideally containing one or more complete sentences. The second step is to run each chunk through something called an embedder which converts a chunk into a vector. In simple terms a vector is a list of numbers that represent the meaning of the chunk. The last step would be to insert the vector representation of the chunk into a data store.

Now that we have ingested the Employee Handbook into our vector data store we can query this data using Natural Language. To do this we first translate the users input to a vector using the same chunk to vector embedder that we used previously to ingest our Employee Handbook. Then we use the user question vector to retrieve the most relevant vectors of our Employee Handbook. Now that we have the relevant chunks of data we can construct a message to send to an LLM. The message we send to the LLM will contain the relevant Employee Handbook chunks along with the users question along with any specific instructions we would like to provide the LLM. The LLM should now be able to respond to the users question with an answer that is of high quality lacking any hallucinations.

**Key points to take into consideration.**

The Ingestion process only needs to be completed once for any specific version of the document or data that is to be used with RAG. As existing content is updated the Ingestion process will need to be ran again to ingest the latest version.

Information that we want to leverage with RAG must be predetermined and ingested prior to user interaction.

One advantage to RAG is that during the querying process you can insert a circuit breaker of sorts. Lets say the user has asked a question that results in only Employee Handbook chunks that have a low correlation to the users question. We can return a default message to the user, or let the LLM know that no quality data was found in regards to the users question and ask it to tell the user it doesn't know the correct answer due to a lack of relevant information.

## 3. What are the components of RAG?

### Loader

The first component of a RAG system is the Loader.

The only job of the Loader is to take the source content and ensure that it is converted into a usable format for the next step in the RAG pipeline. For almost all workflows you are looking to transform the input into raw text.

Most RAG systems will have specific Loaders tailored to different input file formats. That means you might have a Loader that is focused on PDFs, another Loader that is used for Microsoft Word files, another Loader that is used for Webpages, and yet another Loader that is used for Databases. What you need to understand is that your source content needs to be converted into a format (usually plain text) that the next step in the RAG workflow can understand.

### Chunker

Now that we have our source content transformed into raw text, we will use the Chunker to break it up into smaller chunks of data. A chunk typically ranges from a few sentences to a few paragraphs.

The goal is to create small chunks of data which roughly represent a single meaning. Having a chunk represent a single meaning will enable us to have more precise vector representation of the chunk. A more precise vector representation of a chunk will enable higher quality results when querying the Vector Store later on during the process.

Overall a good Chunker will break the content up into logical chunks, meaning we don't want the Chunker to split on a dot in .com of a URL or in the middle of a sentence.

There are many different strategies for chunking, including but not limited to, fixed size (N words/characters), sentence boundaries, paragraph boundaries, or grouping by topic.

There is also the concept of Chunk Overlap, where chunks may intentionally share some sentences with adjacent chunks with the goal of not losing context when querying data later.

Later on we will do some experimentation to see if we can find any interesting data points around things like strategy and chunk size.

Right now you need to understand that the general idea is that we use the Chunker to break the long source material into more manageable blocks that we will later convert into Vectors.

### Embedder

The Embedder, sometimes also called an Embedding Model, will take the chunks of content and convert the text into a Vector. A vector mathematically represents the _semantic meaning_ of the text. The goal here is to assign vector values that are close to the vector values of chunks with similar meaning. For example, "If you see something that looks to be a security concern, please call the 24/7 security hotline" and "The security hotline number is 555-555-1234" should have numerically close vector values.

This close or far numerical value of the vectors is what enables us to search the Vector Store later on when querying for relevant data to a user's question. We are searching the Vector Store for similar _meaning_.

Hopefully now you can see why it is important that we use the same Embedder for the Ingestion step and also the Querying step of the RAG workflow.

### Vector Store

The Vector Store aka Vector Database is the home of all of your chunk vectors.

Its job is to provide N most similar vectors to a given input. e.g. "Given this query vector, what are the N most similar vectors in the database?" The underlying process for this is called Approximate Nearest Neighbor (ANN) search.

There are many choices when it comes to Vector Stores and we will be exploring many options later on in this series. It is possible to leverage NumPy as a Vector Store, but it is recommended to use something like pgvector (PostgreSQL extension that adds Vector Storage functionality) or Weaviate, which offers a fully managed product along side an open source option if you would like to self host. All in all there is a lot of competition in this market so you will have plenty of options to choose from.

What is important to understand right now is that Vectors are used to represent the _meaning_ of a Chunk and you need a 'specialized' system to store and query the vectors.

### Retriever

The Retriever takes care of the querying of the vector store. You provide the Retriever with the user's question, the Retriever runs the user's question through the embedder, and then queries the Vector Store for the top N most relevant Chunks.

This is also a place in the workflow where you can implement additional filtering and re-ranking of the returned data. For example you could implement a filter that ensures that only content that the specific user has permissions to see is provided to the LLM.

Conceptually the Retriever sits between the user's question and the Vector Store, returning the most relevant chunks to be passed along to the LLM.

### LLM

The Large Language Model (LLM) is the final step in the RAG workflow. The LLM has already been trained on loads of information but is lacking specific knowledge around the required workflow. We provide the LLM with the user's question, relevant data from the Vector Store, and any additional information we would like e.g. "Only provide answers based on the provided content". The LLM's job is to generate a natural language response that is provided back to the user.

There are many options for LLMs nowadays. We will play with LLMs that you can run locally, LLMs that you self host in the cloud, and fully managed LLMs that are provided by companies like Anthropic and OpenAI.

Leveraging the RAG workflow will enable the LLM to respond with answers grounded in truth.

## 4. Options for each component

To give some visibility to the options available for the different components of a RAG workflow I have added some options below. This isn't an exhaustive list but should act as a good starting point for exploring the options available.

### Loader Options

 * [LangChain Document Loader](https://docs.langchain.com/oss/python/integrations/document_loaders)
 * [LlamaIndex](https://github.com/run-llama/llama_index)
 * [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
 * [Unstructured](https://github.com/Unstructured-IO/unstructured)

### Chunker Options

 * [LangChain RecursiveCharacterTextSplitter](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter)
 * [LlamaIndex Node Parsers](https://developers.llamaindex.ai/python/framework/module_guides/loading/node_parsers/)
 * [spaCy](https://spacy.io/usage/linguistic-features)
 * [NLTK](https://www.nltk.org/howto/chunk.html)
 * Semantic Chunking - Newer approach where an embedding model is used to detect topic shifts and split there instead of by size

### Embedder Options

#### Local / Self-Hosted
 * [nomic-embed-text](https://www.nomic.ai/news/nomic-embed-text-v1)
 * [all-miniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
 * [mxbai-embed-large](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)

#### Hosted / API
 * [OpenAI (text-embedding-3-small / text-embedding-3-large)](https://developers.openai.com/api/docs/guides/embeddings/)
 * [Cohere Embed](https://docs.cohere.com/docs/embeddings)
 * [Google Text Embedding Models](https://ai.google.dev/gemini-api/docs/embeddings)

### Vector Store Options

#### In-memory / file-based
 * [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
 * [ChromaDB](https://github.com/chroma-core/chroma/)
 * [NumPy](https://github.com/numpy/numpy)

#### Self Hosted
 * [Weaviate](https://github.com/weaviate/weaviate)
 * [Qdrant](https://github.com/qdrant/qdrant)
 * [Milvus](https://github.com/milvus-io/milvus)
 * [pgvector](https://github.com/pgvector/pgvector)

#### Fully Managed
 * [Weaviate Cloud](https://weaviate.io)
 * [Qdrant Cloud](https://qdrant.tech)
 * [Pinecone](https://www.pinecone.io)
 * [MongoDB Atlas Vector Search](https://www.mongodb.com/products/platform/atlas-vector-search)

### Retriever Options

 * Basic similarity search (Usually provided by Vector Store)
 * [LangChain](https://www.langchain.com/retrieval)
 * [LlamaIndex](https://developers.llamaindex.ai/python/framework/module_guides/querying/retriever/)
 * Hybrid search - Combines vector search (semantic) with traditional keyword (BM25) search
 * Re-rankers - a second model that re-scores the initial results for higher precision
   - [Cohere Rerank](https://cohere.com/rerank)

### LLM Options

#### Locally Hosted

For this project we will be leveraging LM Studio to manage and run our locally hosted models.

 * [Qwen3.5](https://github.com/QwenLM/Qwen3.5)
 * [OpenAI OSS](https://github.com/openai/gpt-oss)
 * [Liquify Models](https://www.liquid.ai/models)

#### Self Hosted

With regards to Self Hosted we will look into leveraging vLLM and or Ollama

For this project we will probably leverage roughly the same Open Weight Models.
With some testing of larger models that won't run on a Macbook Air M3 16GB

#### Fully Managed / API
There are many options in this part of the workflow. For simplicity we will probably play with one or two providers.

 * [Anthropic](https://www.anthropic.com)
 * [OpenAI](https://openai.com)
 * [OpenRouter](https://openrouter.ai) (Interface to multiple Fully Managed LLM providers)

## 5. Key Trade Offs

### Cost

Costs will show up at almost every point in a RAG workflow. Even when hosting locally you need to consider that the device you are running them on costs some form of money. It might be possible to run a full RAG workflow with Free Tier only plans but you will eventually run into usage limits.

**Embedding:** Fully Managed or API providers will charge per token. Local or Self Hosted options are free but still require hardware to run the tools on.

**Vector Store:** Managed solutions might offer a Free Tier but eventually you will need to start paying for storage and queries. Local and Self Hosted options will still run into costs for hardware. In general you can think of Managed solutions as higher cost but lower setup effort, while Local or Self Hosted options as cheaper but require more work to setup, configure, and manage.

**LLM:** RAG workflows with heavy usage will probably find that the LLM is the highest cost. A benefit to using RAG is that the system is designed to optimize token (Context Window) usage which will help vs including the complete dataset in your prompt. But no matter how you configure and operate your workflows the LLM part will likely be the highest cost point. Locally hosted LLM models tend to be slow and of lower quality due to limitations around Memory and Compute. Self Hosting an LLM can be surprisingly expensive, at first the price per hour might seem low but when you consider the total monthly cost to have the system running 24/7 the price can become eye watering. The LLM will probably be the first place you start to consider leveraging Third Party APIs as you only pay for usage.

### Privacy

Privacy is a very sensitive subject when it comes to AI workflows. Most AI workflows that people think of require sending your data to a third party system for processing. This can become a major issue when we start to think about sensitive data like medical records, legal documents, financial information, etc. Some industries have regulations on how data is stored and handled, which might require that data isn't sent to third party systems.

Some of these challenges can be addressed with Local and Self Hosted workflows, allowing you to have total control over where and how your data is processed and stored.

The main thing to remember here is that any time you leverage a third party API or Fully Managed service your data is leaving your 'walled garden' and is effectively no longer in your direct control. Some workflows this might be totally fine and in other workflows it might be a deal breaker. It is up to you to understand the risks associated with leveraging third party systems for any form of data processing or storage.

### Scalability

When we start to think about scalability there are many factors that need to be considered. Tooling, Hardware, Storage, etc. Every part of a system will have some scaling dynamic associated with it.

When running a RAG workflow locally you effectively have no options for scaling. The hardware is what it is and short of replacing the underlying components of the computer you are stuck with the current level of performance.

From the Self Hosted standpoint if you need to increase the performance of a section of the tool chain you have the ability to upgrade to more performant hardware. But this does come at the challenge of needing to perform the system migration yourself. This added complexity is what drives most people to moving to a fully managed solution.

Scalability is where Fully Managed solutions really shine. You are paying a premium to offload the work associated with managing your own infrastructure but in exchange you get to have a system that automatically scales without your direct intervention.

### Complexity

The level of complexity that you deal with will be partially dependent on your workflow hosting methods. Fully managed workflows offer the lowest operational complexity while usually having the highest cost and possibly limited control over how the system operates. Self Hosted workflows tend to fall into the medium to high complexity camp but allow for complete control over the workflow. Locally hosted workflows tend to be less complex to setup than a Self Hosted workflow but you are limited to the scale that is achievable.

As you explore more advanced RAG patterns you might find that there is a trade off between increased quality and system complexity. Everyone will have a different sweet spot when it comes to complexity vs works well enough. This sweet spot can only be identified through experimentation and testing.

## 6. What "naive RAG" looks like vs more advanced patterns

### Naive RAG

**Ingestion:** Load -> Chunk -> Embed -> Store
**Querying:** User Question -> Embed -> Vector Search -> Top N Chunks -> LLM Prompt -> Answer

When we think about the most simple form of RAG we have what is listed above. No special logic, no quality checks, no optimization. For simple use cases it is just fine. For our example of the Employee Handbook search it should fill that need without issue.

With this simple system we might run into problems due to simple chunking strategies resulting in vectors that cover multiple topics. Fixed Top N result size returned from the Vector Store might yield an outlier or two that don't match the semantics of the user's question leading to the LLM being fed with irrelevant content. Leveraging a static prompt structure to the LLM might be too constrained for optimal responses for different types of questions.

On the surface this basic RAG workflow might look sufficient but as we look closer at the individual parts we start to see limitations and challenges.

### More Advanced Patterns

There are many modifications that can be made to RAG workflows that can increase the quality and accuracy of the system. Some of these modifications are simple and easy to understand while others are more complicated and difficult to implement. Below are some examples of enhancements that can be implemented in your RAG workflow.

**Query rewriting:** Before searching the Vector Store, you could pass the user's question off to an LLM to rewrite or expand on the user's question with the goal of improving the quality of the response from the Vector Store. This does come at the cost of an additional LLM call which will increase costs and overall response time of the RAG workflow.

**HyDE (Hypothetical Document Embedding):** Instead of embedding the user's question directly, an LLM generates a hypothetical ideal answer, and that gets sent to the Vector Store. The loose idea is that an answer shaped vector will yield better results than a question shaped vector.

**Re-Ranking:** Once we get a response back from the Vector Store we leverage another model to score and re-order the results by relevance before passing things off to the LLM. This can help fix times when non relevant chunks have been returned by the Vector Store.

**Multi-hop / Agentic RAG:** This is a system that performs multiple queries against the Vector Store. Iterating on its query in an attempt to find additional information that might be relevant to the user's question. This process is helpful for complex or multi-part questions.

**Hybrid Search:** Combines vector search with traditional keyword search. This is good for situations where exact terminology matters but semantic search underperforms.

**Contextual Compression:** When dealing with larger Chunk sizes you will tend to end up with sentences which are irrelevant to the user's question. The idea behind Contextual Compression is that the Chunk is processed to ensure that only relevant data is sent to the LLM, removing unnecessary data.

**Relevance Filtering / Guardrails:** Leveraging filters to ensure that the LLM is only provided with relevant information when being asked to answer the user's question. We can also include guardrails in the form of a 'circuit breaker' where if no relevant chunks were returned by the Vector Store the system works to provide a user with an "I don't know" response.

Right now the main thing to understand is that a 'naive RAG' is a great starting point and might be the perfect solution to your requirements. You will only be able to understand what improvements are needed by building out a basic system and iterating on it until you land on a sufficient solution to the workflow.
