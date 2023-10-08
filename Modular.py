import pyserini
import math
from pyserini.index.lucene import IndexReader
from pyserini.search import get_topics,LuceneSearcher

# def configure_java_environment():
#     import os
#     os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

def retrieve_topics(dataset: str):
    topics = get_topics(dataset)
    return topics

def search_for_query(index: str, query: str, num_results: int = 10):
    searcher = LuceneSearcher.from_prebuilt_index(index)
    hits = searcher.search(query, num_results)
    return hits

def extract_documents_from_hits(hits, num_results: int = 10):
    import json
    documents = []
    for i in range(num_results):
        jsondoc = json.loads(hits[i].raw)
        documents.append(jsondoc["contents"][:1000])
    return documents

def process_documents(documents):
    tokenized_documents = [doc.split() for doc in documents]
    preprocessed_documents = [' '.join(doc) for doc in tokenized_documents]
    return preprocessed_documents

def calculate_idf(word: str, index: str):
    indexer = IndexReader.from_prebuilt_index(index)
    total_documents = indexer.stats()["documents"]
    df, cf = indexer.get_term_counts(word)
    idf = math.log(total_documents / (df + 1))
    return idf

def vectorize_documents_tfidf(documents, use_idf=False, stop_words=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=use_idf)
    tfidf = tfidf_vectorizer.fit_transform(documents)
    return tfidf


# def dice_counterfactual_explanations():
#     pass

# def google_drive_tasks():
#     pass

topics = retrieve_topics('msmarco-passage-dev-subset')
hits = search_for_query('msmarco-passage', 'average rent in california', 30)
documents = extract_documents_from_hits(hits, 30)
processed_docs = process_documents(documents)
tfidf = vectorize_documents_tfidf(processed_docs, use_idf=False)

