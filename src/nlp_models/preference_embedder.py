import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

class PreferenceEmbedder:
    """
    Manages the generation of contextual embeddings for predefined attributes
    and calculates the similarity of review texts to these attributes.
    """
    def __init__(self, model_name: str = 'google-bert/bert-base-multilingual-cased'): # Changed model to mBART
        """
        Initializes the SentenceTransformer model.

        Args:
            model_name (str): The name of the pre-trained SentenceTransformer model to use.
                              'facebook/mbart-large-cc25' is a multilingual mBART model.
        """
        print(f"Loading SentenceTransformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully.")
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

# Example Usage (for testing/demonstration)
if __name__ == "__main__":
    # Define the expanded list of predefined attributes
    predefined_attributes = [
        "Cozy", "Trendy", "Romantic", "Lively", "Quiet", "Elegant", "Casual", "artistic",
        "Bohemian", "Family-Friendly", "Pet-Friendly", "Outdoor Seating", "Good for Groups",
        "Good for Solo", "Gourmet", "Comfort Food", "Healthy", "Vegan-Friendly", "Dessert",
        "Coffee", "Date", "Scenic View", "Parking Available", "Wheelchair Accessible",
        "Wi-Fi Available", "Workspace"
    ]

    # Initialize the embedder with the mBART model
    embedder = PreferenceEmbedder(model_name='facebook/mbart-large-cc25')

    # Generate embeddings for the attributes
    embedder.generate_attribute_embeddings(predefined_attributes)

    # Test with some example review texts
    review1 = "This cafe has such a warm and inviting atmosphere, perfect for reading."
    review2 = "Super modern decor and great music, definitely a trendy spot for young people."
    review3 = "The dim lighting and soft jazz made it ideal for a romantic dinner."
    review4 = "Very noisy and crowded, but the energy was amazing! Great for groups."
    review5 = "It's a bit loud, but the food was fantastic. Not very cozy though."
    review6 = "" # Empty review
    review7 = "This place is both trendy and cozy, a rare combination!"
    review8 = "Excellent fine dining experience with an elegant ambiance and gourmet dishes."
    review9 = "A great spot for remote work, good coffee and reliable Wi-Fi."


    print("\n--- Review Attribute Scores ---")
    print(f"Review 1 ('{review1}'): {embedder.get_review_attribute_scores(review1)}")
    print(f"Review 2 ('{review2}'): {embedder.get_review_attribute_scores(review2)}")
    print(f"Review 3 ('{review3}'): {embedder.get_review_attribute_scores(review3)}")
    print(f"Review 4 ('{review4}'): {embedder.get_review_attribute_scores(review4)}")
    print(f"Review 5 ('{review5}'): {embedder.get_review_attribute_scores(review5)}")
    print(f"Review 6 ('{review6}'): {embedder.get_review_attribute_scores(review6)}")
    print(f"Review 7 ('{review7}'): {embedder.get_review_attribute_scores(review7)}")
    print(f"Review 8 ('{review8}'): {embedder.get_review_attribute_scores(review8)}")
    print(f"Review 9 ('{review9}'): {embedder.get_review_attribute_scores(review9)}")
