from collections import defaultdict
import random
import os
from bs4 import BeautifulSoup
import time
from matplotlib import pyplot as plt

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

    def rolling_hash(self, shingle):
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
            shingles.add(self.rolling_hash(shingle))
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

    def _make_hash_function(self, s, universe_size):
        """
        Creates a hash function with random coefficients.

        Args:
        universe_size (int): The universe size for the hash function.

        Returns:
        function: A hash function.
        """
        random.seed(s)
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
def process_documents(docs, shingler, minhasher, threshold):
    # Shingling
    shingled_docs = [shingler.shingle_document(doc) for doc in docs]

    # MinHashing
    signatures = [minhasher.create_signature(shingles) for shingles in shingled_docs]

    lsh = LSH(num_bands=10, threshold=threshold)
    for signature in signatures:
        lsh.add_signature(signature)

    similar_pairs = lsh.find_similar()
    return similar_pairs

def evaluation(threshold, k):
    # Load the articles
    file_path = os.path.join("extracted_files", "reut2-000.sgm")
    articles = parse_sgm(file_path)
    print("Total number of documents:", len(articles))
    print(f"Shingle size: {k}, similarity threshold: {threshold}")
    # Initialize the classes
    shingler = Shingling(k=5)
    minhasher = MinHashing(n=100, universe_size=10000)

    total_len = len(articles)

    # Calculate dynamic batch sizes
    batch_sizes = np.linspace(0, total_len, 7, dtype=int)[1:]

 # Lists to store batch sizes and corresponding runtimes
    batch_sizes_list = []
    runtimes_list = []
    num_items = []

    # Evaluate runtime for different batch sizes
    for batch_size in batch_sizes:
        start_time = time.time()
        items = process_documents(articles[:batch_size], shingler, minhasher, threshold)
        end_time = time.time()
        runtime = end_time - start_time

        print(f"Runtime for {batch_size} articles: {runtime:.2f} seconds")
        print(f"Similar items found: {len(items)}")
        print(f"-----------------------------------------------------------")

        # Store batch size and runtime for plotting
        batch_sizes_list.append(batch_size)
        runtimes_list.append(runtime)

    # Plotting
    plt.plot(batch_sizes_list, runtimes_list, marker='o')
    plt.title('Runtime as a Function of Documents')
    plt.xlabel('Number of documents')
    plt.ylabel('Runtime (seconds)')
    plt.show()


# print("Set the shingle size: ")
# k = input()

# print("Set the threshold: ")
# threshold = input()

evaluation(threshold=0.8, k=5)
