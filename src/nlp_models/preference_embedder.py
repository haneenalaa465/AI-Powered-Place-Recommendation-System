import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

class PreferenceEmbedder:
    """
    Manages the generation of contextual embeddings for predefined attributes
    and calculates the similarity of review texts to these attributes.
    """
    def __init__(self, model_name: str = 'jinaai/jina-embeddings-v3'):
        """
        Initializes the SentenceTransformer model.

        Args:
            model_name (str): The name of the pre-trained SentenceTransformer model to use.
        """
        print(f"Loading SentenceTransformer model: {model_name}...")
        try:
            # Added trust_remote_code=True to handle potential custom components
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Falling back to 'paraphrase-multilingual-MiniLM-L12-v2' for stability.")
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("Fallback model 'paraphrase-multilingual-MiniLM-L12-v2' loaded successfully.")
        
        self.attribute_embeddings = None
        self.attribute_names = None

    def generate_attribute_embeddings(self, attributes: list[str]) -> np.ndarray:
        """
        Generates and stores embeddings for a list of predefined attributes.

        Args:
            attributes (list[str]): A list of strings representing the predefined attributes.

        Returns:
            np.ndarray: A 2D numpy array where each row is the embedding for an attribute.
        """
        if not attributes:
            raise ValueError("Attribute list cannot be empty.")

        print(f"Generating embeddings for {len(attributes)} attributes...")
        self.attribute_embeddings = self.model.encode(attributes, convert_to_numpy=True)
        self.attribute_names = attributes
        print("Attribute embeddings generated.")
        return self.attribute_embeddings

    def get_review_attribute_scores(self, review_text: str) -> dict[str, float]:
        """
        Calculates the similarity of a given review text to the predefined attributes.
        Requires `generate_attribute_embeddings` to have been called first.

        Args:
            review_text (str): The text of the review.

        Returns:
            dict[str, float]: A dictionary where keys are attribute names and values
                              are their cosine similarity scores (0.0 to 1.0) with the review.
                              Returns an empty dict if attributes are not set or review is empty.
        """
        if self.attribute_embeddings is None or self.attribute_names is None:
            print("Warning: Attributes not set. Call generate_attribute_embeddings first.")
            return {}
        if not review_text or not review_text.strip():
            # Return 0 for empty or whitespace-only reviews for all attributes
            return {attr: 0.0 for attr in self.attribute_names} 

        # Generate embedding for the review text
        review_embedding = self.model.encode([review_text], convert_to_numpy=True)

        # Calculate cosine similarity between review embedding and all attribute embeddings
        # The result will be a 1xN array, where N is the number of attributes
        similarities = cosine_similarity(review_embedding, self.attribute_embeddings)[0]

        # Map similarity scores back to attribute names
        attribute_scores = {
            self.attribute_names[i]: float(similarities[i])
            for i in range(len(self.attribute_names))
        }
        return attribute_scores

# Example Usage (for testing/demonstration of PreferenceEmbedder only)
if __name__ == "__main__":
    # Define a subset of predefined attributes for quick testing of embedder
    predefined_attributes_subset = ["Cozy", "Trendy", "Romantic", "Quiet", "Workspace"]

    # Initialize the embedder
    embedder = PreferenceEmbedder(model_name='jinaai/jina-embeddings-v3')

    # Generate embeddings for the attributes
    embedder.generate_attribute_embeddings(predefined_attributes_subset)

    print("\n--- Individual Review Attribute Scores ---")
    print(f"Review EN1 ('This cafe has such a warm and inviting atmosphere.'): {embedder.get_review_attribute_scores('This cafe has such a warm and inviting atmosphere.')}")
    print(f"Review AR1 ('المكان رايق والقاعدة بتاعته حلوة.'): {embedder.get_review_attribute_scores('المكان رايق والقاعدة بتاعته حلوة.')}")
    print(f"Review FR1 ('Idéal pour travailler, le Wi-Fi est rapide.'): {embedder.get_review_attribute_scores('Idéal pour travailler, le Wi-Fi est rapide.')}")
