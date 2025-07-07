import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

class PreferenceEmbedder:
    """
    Manages the generation of contextual embeddings for predefined attributes
    and calculates the similarity of review texts to these attributes.
    """
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initializes the SentenceTransformer model.

        Args:
            model_name (str): The name of the pre-trained SentenceTransformer model to use.
                              'paraphrase-multilingual-MiniLM-L12-v2' is a good multilingual
                              model optimized for sentence similarity.
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

    # Initialize the embedder with the recommended multilingual model
    embedder = PreferenceEmbedder(model_name='paraphrase-multilingual-MiniLM-L12-v2')

    # Generate embeddings for the attributes
    embedder.generate_attribute_embeddings(predefined_attributes)

    # Test with some example review texts (English, Arabic, Tunisian, French)
    print("\n--- Review Attribute Scores ---")

    # English Examples (retained)
    review_en1 = "This cafe has such a warm and inviting atmosphere, perfect for reading."
    review_en2 = "Super modern decor and great music, definitely a trendy spot for young people."
    review_en3 = "The dim lighting and soft jazz made it ideal for a romantic dinner."
    review_en4 = "Very noisy and crowded, but the energy was amazing! Great for groups."
    review_en5 = "It's a bit loud, but the food was fantastic. Not very cozy though."
    review_en6 = "" # Empty review
    review_en7 = "This place is both trendy and cozy, a rare combination!"
    review_en8 = "Excellent fine dining experience with an elegant ambiance and gourmet dishes."
    review_en9 = "A great spot for remote work, good coffee and reliable Wi-Fi."

    print(f"\n--- English Reviews ---")
    print(f"Review EN1 ('{review_en1}'): {embedder.get_review_attribute_scores(review_en1)}")
    print(f"Review EN2 ('{review_en2}'): {embedder.get_review_attribute_scores(review_en2)}")
    print(f"Review EN3 ('{review_en3}'): {embedder.get_review_attribute_scores(review_en3)}")
    print(f"Review EN4 ('{review_en4}'): {embedder.get_review_attribute_scores(review_en4)}")
    print(f"Review EN5 ('{review_en5}'): {embedder.get_review_attribute_scores(review_en5)}")
    print(f"Review EN6 ('{review_en6}'): {embedder.get_review_attribute_scores(review_en6)}")
    print(f"Review EN7 ('{review_en7}'): {embedder.get_review_attribute_scores(review_en7)}")
    print(f"Review EN8 ('{review_en8}'): {embedder.get_review_attribute_scores(review_en8)}")
    print(f"Review EN9 ('{review_en9}'): {embedder.get_review_attribute_scores(review_en9)}")

    # Arabic Examples
    review_ar1 = "هذا المقهى دافئ ومريح جداً، مثالي للقراءة." # This cafe is very warm and comfortable, ideal for reading. (Cozy, Quiet)
    review_ar2 = "ديكور عصري وموسيقى رائعة، مكان أنيق للشباب." # Modern decor and great music, a stylish place for young people. (Trendy)
    review_ar3 = "الإضاءة الخافتة والموسيقى الهادئة جعلته مثالياً لعشاء رومانسي." # Dim lighting and quiet music made it ideal for a romantic dinner. (Romantic)
    review_ar4 = "المكان صاخب ومزدحم جداً، لكن الطاقة كانت مذهلة! رائع للمجموعات." # The place is very noisy and crowded, but the energy was amazing! Great for groups. (Lively, Good for Groups)
    review_ar5 = "أكلهم بنين برشا والجو مزيان." # Their food is very tasty and the atmosphere is nice. (Tunisian dialect - "بنين برشا" for very tasty, "مزيان" for nice/beautiful) - (Comfort Food, Casual)
    review_ar6 = "بلاصة نظيفة و فيها إنترنت قوية تنجم تخدم على روحك." # A clean place with strong internet, you can work comfortably. (Tunisian dialect - "بلاصة" for place, "نظيفة" for clean, "قوية" for strong, "تنجم تخدم على روحك" for you can work comfortably) - (Workspace, Wi-Fi Available)


    print(f"\n--- Arabic Reviews ---")
    print(f"Review AR1 ('{review_ar1}'): {embedder.get_review_attribute_scores(review_ar1)}")
    print(f"Review AR2 ('{review_ar2}'): {embedder.get_review_attribute_scores(review_ar2)}")
    print(f"Review AR3 ('{review_ar3}'): {embedder.get_review_attribute_scores(review_ar3)}")
    print(f"Review AR4 ('{review_ar4}'): {embedder.get_review_attribute_scores(review_ar4)}")
    print(f"Review AR5 (Tunisian) ('{review_ar5}'): {embedder.get_review_attribute_scores(review_ar5)}")
    print(f"Review AR6 (Tunisian) ('{review_ar6}'): {embedder.get_review_attribute_scores(review_ar6)}")

    # French Examples
    review_fr1 = "Ce café a une ambiance tellement chaleureuse et accueillante, parfait pour lire." # This cafe has such a warm and inviting atmosphere, perfect for reading. (Cozy, Quiet)
    review_fr2 = "Décor super moderne et bonne musique, un endroit vraiment branché pour les jeunes." # Super modern decor and good music, definitely a trendy spot for young people. (Trendy)
    review_fr3 = "L'éclairage tamisé et le jazz doux en ont fait l'endroit idéal pour un dîner romantique." # The dim lighting and soft jazz made it ideal for a romantic dinner. (Romantic)
    review_fr4 = "Très bruyant et bondé, mais l'énergie était incroyable ! Idéal pour les groupes." # Very noisy and crowded, but the energy was amazing! Great for groups. (Lively, Good for Groups)
    review_fr5 = "La nourriture était saine et délicieuse, avec de nombreuses options végétaliennes." # The food was healthy and delicious, with many vegan options. (Healthy, Vegan-Friendly)
    review_fr6 = "Vue imprenable et service impeccable, parfait pour une date." # Breathtaking view and impeccable service, perfect for a date. (Scenic View, Date)


    print(f"\n--- French Reviews ---")
    print(f"Review FR1 ('{review_fr1}'): {embedder.get_review_attribute_scores(review_fr1)}")
    print(f"Review FR2 ('{review_fr2}'): {embedder.get_review_attribute_scores(review_fr2)}")
    print(f"Review FR3 ('{review_fr3}'): {embedder.get_review_attribute_scores(review_fr3)}")
    print(f"Review FR4 ('{review_fr4}'): {embedder.get_review_attribute_scores(review_fr4)}")
    print(f"Review FR5 ('{review_fr5}'): {embedder.get_review_attribute_scores(review_fr5)}")
    print(f"Review FR6 ('{review_fr6}'): {embedder.get_review_attribute_scores(review_fr6)}")
