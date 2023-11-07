from collections import defaultdict
import random
import os
from bs4 import BeautifulSoup
import time

import numpy as np
from collections import defaultdict
from itertools import combinations


class Shingling:
    """
    A class used to create shingles from a given document.
    
    Attributes:
    k (int): Length of each shingle.
    modulo (int): A large prime number used for modulo operation in hash function.
    """
    def __init__(self, k, modulo=1000000007):
        self.k = k
        self.modulo = modulo

    def _rolling_hash(self, shingle):
        """
        Generates a hash value for a given shingle.
        
        Args:
        shingle (str): The shingle to hash.

        Returns:
        int: The hash value of the shingle.
        """
        hash_value = 0
        for i, c in enumerate(shingle):
            hash_value = (hash_value + (ord(c) * (31 ** i))) % self.modulo
        return hash_value

    def shingle_document(self, document):
        """
        Creates a set of hashed shingles from a given document.
        
        Args:
        document (str): The document to shingle.

        Returns:
        set: A set of hashed shingles.
        """
        shingles = set()
        for i in range(len(document) - self.k + 1):
            shingle = document[i:i + self.k]
            shingles.add(self._rolling_hash(shingle))
        return set(sorted(shingles))

    def get_text_shingles(self, document):
        """
        Creates a set of text shingles from a given document.
        
        Args:
        document (str): The document to shingle.

        Returns:
        set: A set of text shingles.
        """
        shingles = set()
        for i in range(len(document) - self.k + 1):
            shingle = document[i:i + self.k]
            shingles.add(shingle)
        return shingles


class CompareSets:
    """
    A class used to compare sets, specifically for computing Jaccard similarity.
    """
    @staticmethod
    def jaccard_similarity(set1, set2):
        """
        Computes the Jaccard similarity between two sets.

        Args:
        set1 (set): The first set.
        set2 (set): The second set.

        Returns:
        float: The Jaccard similarity between the two sets.
        """
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union


class MinHashing:
    """
    A class used to create MinHash signatures for sets of shingles.
    
    Attributes:
    n (int): The number of hash functions.
    hash_functions (list): A list of hash functions.
    """
    def __init__(self, n, universe_size):
        self.n = n
        self.hash_functions = [self._make_hash_function(p, universe_size) for p in range(n)]

    def _make_hash_function(self, seed, universe_size):
        """
        Creates a hash function with random coefficients.

        Args:
        seed (int): A seed value for random number generation.
        universe_size (int): The universe size for the hash function.

        Returns:
        function: A hash function.
        """
        a = random.randint(1, universe_size)
        b = random.randint(0, universe_size)
        return lambda x: (a * hash(x) + b) % universe_size

    def create_signature(self, shingle_set):
        """
        Creates a MinHash signature for a set of shingles.

        Args:
        shingle_set (set): The set of shingles to hash.

        Returns:
        list: A MinHash signature.
        """
        signature = []
        for h in self.hash_functions:
            min_hash = min([h(shingle) for shingle in shingle_set], default=0)
            signature.append(min_hash)
        return signature


class CompareSignatures:
    """
    A class used to compare MinHash signatures.
    """
    @staticmethod
    def signature_similarity(sig1, sig2):
        """
        Computes the similarity between two MinHash signatures.

        Args:
        sig1 (list): The first MinHash signature.
        sig2 (list): The second MinHash signature.

        Returns:
        float: The proportion of indices where the signatures match.
        """
        matches = sum(1 for i, j in zip(sig1, sig2) if i == j)
        return matches / len(sig1)


class LSH:
    def __init__(self, num_bands, threshold):
        self.num_bands = num_bands
        self.threshold = threshold
        self.buckets = defaultdict(list)
        self.signatures = []
    
    def _hash_band(self, band):
        """
        Hashes a band (a tuple) of a signature.
        For simplicity, we're using the built-in hash function.
        """
        return hash(band)
    
    def add_signature(self, signature):
        """
        Adds a signature to the LSH object and hashes its bands.
        """
        self.signatures.append(signature)
        idx = len(self.signatures) - 1
        num_rows = int(len(signature) / self.num_bands)
        
        # Hash each band of the signature and add the index to the corresponding bucket
        for band_idx in range(self.num_bands):
            start = band_idx * num_rows
            end = (band_idx + 1) * num_rows
            band = tuple(signature[start:end])
            hashed_band = self._hash_band(band)
            print(f" hashed band: {hashed_band}")
            self.buckets[hashed_band].append(idx)
    
    def find_candidates(self):
        """
        Finds candidate pairs of signatures that agree on at least a fraction t of their components.
        """
        candidates = set()
        for bucket in self.buckets.values():
            if len(bucket) > 1:
                for pair in combinations(bucket, 2):
                    candidates.add(pair)
        return candidates
    
    def find_similar(self):
        """
        After finding candidate pairs, this method verifies if they agree on at least a fraction t of their components.
        """
        candidates = self.find_candidates()
        similar_pairs = set()
        for i, j in candidates:
            sig1, sig2 = self.signatures[i], self.signatures[j]
            similarity = sum(1 for x, y in zip(sig1, sig2) if x == y) / len(sig1)
            if similarity >= self.threshold:
                similar_pairs.add((i, j))
        return similar_pairs



# Function to parse SGM file and extract text
def parse_sgm(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        soup = BeautifulSoup(file, 'html.parser')
        texts = [reuters.body.get_text() for reuters in soup.find_all('reuters') if reuters.body]
        return texts
    
# Function to process documents through all stages
def process_documents(docs, shingler, minhasher, lsh):
    # Shingling
    shingled_docs = [shingler.shingle_document(doc) for doc in docs]

    # MinHashing
    signatures = [minhasher.create_signature(shingles) for shingles in shingled_docs]

    # LSH
    similar_pairs = lsh.find_similar(signatures)
    return similar_pairs

def evaluation():
    # Load the articles
    file_path = os.path.join("extracted_files", "reut2-000.sgm")
    articles = parse_sgm(file_path)

    # Initialize the classes
    shingler = Shingling(k=10)
    minhasher = MinHashing(n=100, universe_size=10000)
    lsh = LSH(band_size=10, threshold=0.5)

    # Define the different batch sizes
    batch_sizes = [10, 50, 100, 150, 200, 250]

    # Evaluate runtime for different batch sizes
    for batch_size in batch_sizes:
        start_time = time.time()
        process_documents(articles[:batch_size], shingler, minhasher, lsh)
        end_time = time.time()
        print(f"Runtime for {batch_size} articles: {end_time - start_time:.2f} seconds")


def general_tests():
    file_path = os.path.join("extracted_files", "reut2-000.sgm")

    articles = parse_sgm(file_path)

    k = 10  # Length of each shingle

    # Shingling
    shingle_length = 10
    shingler = Shingling(shingle_length)
    shingled_docs = [shingler.shingle_document(doc) for doc in articles[:250]]

    comparer = CompareSets()
    res = comparer.jaccard_similarity(shingled_docs[0], shingled_docs[1])

    print(res)

    # MinHashing
    signature_length = 100
    universe_size = 1000000
    minhasher = MinHashing(signature_length, universe_size)
    signatures = [minhasher.create_signature(shingles) for shingles in shingled_docs]

    compare_sig = CompareSignatures()
    # sig_res = compare_sig.signature_similarity(signatures[4], signatures[16])
    # print(sig_res)


    # LSH 
    band_size = 10
    threshold = 0.00000000000001
    lsh = LSH(band_size, threshold)
    similar_pairs = lsh.find_similar(signatures)

    # Display similar pairs
    print("Similar Document Pairs:")
    for pair in similar_pairs:
        doc1, doc2 = pair
        jaccard_sim = comparer.jaccard_similarity(shingled_docs[doc1], shingled_docs[doc2])
        minhash_sim = compare_sig.signature_similarity(signatures[doc1], signatures[doc2])

        print(f"Document {doc1} is similar to Document {doc2}")
        print(f"Jaccard Similarity (Shingle Sets): {jaccard_sim:.2f}")
        print(f"MinHash Signature Similarity: {minhash_sim:.2f}")

    # text1= shingler.get_text_shingles(articles[29])
    # text2= shingler.get_text_shingles(articles[52])

    # print(articles[29])
    # print("-----------------")
    # print(articles[52])


# print(text1.intersection(text2))

#evaluation()

file_path = os.path.join("extracted_files", "reut2-000.sgm")
articles = parse_sgm(file_path)
k = 10  # Length of each shingle
# Shingling
shingle_length = 10
shingler = Shingling(shingle_length)
shingled_docs = [shingler.shingle_document(doc) for doc in articles]

comparer = CompareSets()
res = comparer.jaccard_similarity(shingled_docs[0], shingled_docs[1])
print(res)

# MinHashing
signature_length = 100
universe_size = 1000000
minhasher = MinHashing(signature_length, universe_size)
signatures = [minhasher.create_signature(shingles) for shingles in shingled_docs]

print(signatures[0])

compare_sig = CompareSignatures()
sig_res = compare_sig.signature_similarity(signatures[0], signatures[1])
print(sig_res)


# LSH 
# band_size = 10
# threshold = 0.5
# lsh = LSH(band_size, threshold)
# similar_pairs = lsh.find_similar([signatures[29], signatures[52]])
# # Display similar pairs
# print("Similar Document Pairs:")
# for pair in similar_pairs:
#     doc1, doc2 = pair
#     jaccard_sim = comparer.jaccard_similarity(shingled_docs[doc1], shingled_docs[doc2])
#     minhash_sim = compare_sig.signature_similarity(signatures[doc1], signatures[doc2])
#     print(f"Document {doc1} is similar to Document {doc2}")
#     print(f"Jaccard Similarity (Shingle Sets): {jaccard_sim:.2f}")
#     print(f"MinHash Signature Similarity: {minhash_sim:.2f}")


    
# Determine the number of bands based on the similarity threshold and number of rows per band
threshold = 0.5
n = len(signatures[0])  # Number of components in a signature
r = int(n * threshold)  # Number of rows per band
b = int(n / r)  # Number of bands

# Initialize LSH
lsh = LSH(num_bands=b, threshold=threshold)

# Add signatures to LSH
for sig in [signatures[859], signatures[874]]:
    lsh.add_signature(sig)

# Find similar pairs
similar_pairs = lsh.find_similar()
print(similar_pairs)