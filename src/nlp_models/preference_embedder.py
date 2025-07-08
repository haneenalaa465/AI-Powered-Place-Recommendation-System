# src/nlp_models/preference_embedder.py

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
    embedder = PreferenceEmbedder(model_name='jinaai/jina-embeddings-v3')

    # Generate embeddings for the attributes
    embedder.generate_attribute_embeddings(predefined_attributes)

    # --- Combined Review Data (English, Arabic, Tunisian, French) ---
    # Note: I've included translations/notes for clarity, these are not part of the review text itself.
    all_reviews_data = {
        "texts": [
            "Absolutely fantastic! The best I've ever had.", # English
            "الأكل كان تحفة والخدمة ممتازة، مكان يستاهل التجربة.", # Arabic: Food was amazing and service excellent, a place worth trying.
            "المكان رايق والقاعدة بتاعته حلوة، أكيد هروح تاني.", # Arabic: The place is calm and its seating is nice, I'll definitely go again.
            "A wonderful experience, highly recommended.", # English
            "شي بيشهي القلب، عنجد بنصحكن فيه.", # Arabic (Levantine): Something that makes the heart crave it, I really recommend it.
            "The food was good, but it was a bit too crowded.", # English
            "المطعم ذا بطل، يستاهل كل ريال.", # Arabic (Gulf): This restaurant is a hero, worth every riyal.
            "شغل عدل ومضبوط.", # Arabic (Egyptian/Levantine): Work is fair and precise.
            "بنة على بنة، يعطيهم الصحة.", # Tunisian: Delicious on delicious, bless them.
            "The service was a little slow.", # English
            "داكشي بنين بزاف، تبارك الله عليهم.", # Moroccan: That thing is very tasty, bless them.
            "المحل واعر والخدمة نقية.", # Moroccan: The place is awesome and the service is clean.
            "Honestly, everything was perfect to the millimeter.", # English
            "كل حاجة كانت مظبوطة.", # Arabic: Everything was precise.
            "الأكل كتير طيب والمنظر بيعقد.", # Arabic (Levantine): The food is very good and the view is amazing.
            "Five stars all the way!", # English
            "مره عجبني المكان، رايق وهادي.", # Arabic: I really liked the place, calm and quiet.
            "It's okay, nothing too special.", # English
            "بلاصة ياسر مزيانة.", # Tunisian: A very nice place.
            "The staff went above and beyond.", # English
            "غادي نرجع ليه مرة أخرى.", # Moroccan: I will return to it again.
            "الخدمة عشرة على عشرة.", # Arabic: Service ten out of ten.
            "The ambiance is perfect for a date night.", # English
            "أحلى قعدة مع رفقاتي.", # Arabic: Best seating with my friends.
            "يستاهل كل ريال تدفعه.", # Arabic: Worth every riyal you pay.
            "المنظر من هناك تحفة.", # Arabic: The view from there is amazing.
            "The food was simply divine.", # English
            "تجربة ممتعة جداً، أنصح به بشدة.", # Arabic: A very enjoyable experience, highly recommend it.
            "المكان جميل جداً والمنظر ساحر.", # Arabic: The place is very beautiful and the view is enchanting.
            "The coffee here is top-notch.", # English
            "كل شيء كان نظيفاً ومنظماً.", # Arabic: Everything was clean and organized.
            "فريق العمل ودود للغاية ومتعاون.", # Arabic: The staff is very friendly and helpful.
            "The prices are reasonable for the quality.", # English
            "اللقمات طيبين والمحل بجنن.", # Arabic (Levantine): The bites are good and the place is amazing.
            "الجلسات الخارجية جميلة جداً.", # Arabic: The outdoor seating is very beautiful.
            "المكان يستحق الزيارة مرة ومرتين وثلاث.", # Arabic: The place is worth visiting once, twice, and thrice.
            "الأكل طيب والأسعار مقبولة.", # Arabic: Food is good and prices are acceptable.
            "The service was impeccable.", # English
            "الخدمة ممتازة وسريعة.", # Arabic: Service is excellent and fast.
            "مكان هادئ ومريح للأعصاب.", # Arabic: A quiet and relaxing place for the nerves.
            "الديكور بسيط وحلو.", # Arabic: The decor is simple and nice.
            "المطعم المفضل عندي.", # Arabic: My favorite restaurant.
            "الجو العام للمكان لطيف.", # Arabic: The overall atmosphere of the place is nice.
            "A hidden gem in the city.", # English
            "القهوة عندهم مضبوطة.", # Arabic: Their coffee is perfect.
            "أنصح فيه للعائلات.", # Arabic: I recommend it for families.
            "موقعه ممتاز وسهل الوصول.", # Arabic: Its location is excellent and easy to access.
            "الطاقم محترف جداً.", # Arabic: The staff is very professional.
            "Incredible value for money.", # English
            "الأكل وصل بسرعة وكان ساخن.", # Arabic: Food arrived quickly and was hot.
            "أجواء المطعم رائعة.", # Arabic: The restaurant's ambiance is wonderful.
            "يستحق كل التقدير.", # Arabic: Deserves all appreciation.
            "تجربة لن أنساها.", # Arabic: An unforgettable experience.
            "Outstanding quality and presentation.", # English
            "كل شي كان مثالي.", # Arabic: Everything was perfect.
            "Perfect in every way.", # English
            "المكان نظيف ومرتب.", # Arabic: The place is clean and tidy.
            "A must-visit spot.", # English
            "الخدمة طيارة.", # Tunisian: Service is fast (like an airplane).
            "ماكلة بنينة على اللخر.", # Tunisian: Food is very tasty.
            "منظر يفتح النفس.", # Arabic: A breathtaking view.
            "الموظفين كانوا بشوشين.", # Arabic: The staff were cheerful.
            "تجربة فريدة من نوعها.", # Arabic: A unique experience.
            "أفضل مكان جربته في حياتي.", # Arabic: The best place I've tried in my life.
            "الأسعار كانت معقولة جداً.", # Arabic: The prices were very reasonable.
            "كل شيء كان على أكمل وجه.", # Arabic: Everything was perfect.
            "المكان يتفوق على المنافسين.", # Arabic: The place surpasses competitors.
            "الديكورات الداخلية مذهلة.", # Arabic: The interior decorations are amazing.
            "خدمة العملاء كانت رائعة.", # Arabic: Customer service was wonderful.
            "A truly memorable meal.", # English
            "الطعم أصلي والنكهات واضحة.", # Arabic: The taste is original and the flavors are clear.
            "المكان فاق توقعاتي.", # Arabic: The place exceeded my expectations.
            "أنصح بزيارته في أقرب فرصة.", # Arabic: I recommend visiting it as soon as possible.
            "الجودة عالية جداً.", # Arabic: The quality is very high.
            "مكان مثالي للاسترخاء.", # Arabic: A perfect place for relaxation.
            "The portions were generous and delicious.", # English
            "العصيرات عندهم طازجة ومنعشة.", # Arabic: Their juices are fresh and refreshing.
            "الموسيقى كانت هادئة ومناسبة.", # Arabic: The music was quiet and suitable.
            "الاهتمام بالتفاصيل كان واضح.", # Arabic: Attention to detail was clear.
            "مستوى النظافة عالي جداً.", # Arabic: The level of cleanliness is very high.
            "كل طبق كان ألذ من الثاني.", # Arabic: Every dish was tastier than the last.
            "I'm already planning my next visit.", # English
            "أحسن اختيار لمناسبة خاصة.", # Arabic: Best choice for a special occasion.
            "The dessert was the highlight of the meal.", # English
            "الموقع استراتيجي ومميز.", # Arabic: The location is strategic and distinctive.
            "فريق عمل سريع الاستجابة.", # Arabic: Responsive staff.
            "الأجواء كانت مريحة.", # Arabic: The atmosphere was comfortable.
            "تجربة تستحق خمس نجوم.", # Arabic: An experience worth five stars.
            "The menu has a great variety of options.", # English
            "مكان تشعر فيه بالترحاب.", # Arabic: A place where you feel welcome.
            "الجودة مقابل السعر ممتازة.", # Arabic: Excellent quality for the price.
            "The best view in town.", # English
            "التقديم كان احترافي.", # Arabic: The presentation was professional.
            "أكيد سأعود مرة أخرى.", # Arabic: I will definitely return again.
            "Ce restaurant est très élégant et les plats sont gastronomiques.", # French: This restaurant is very elegant and the dishes are gourmet.
            "L'ambiance est décontractée, parfaite pour une soirée entre amis.", # French: The ambiance is casual, perfect for a night out with friends.
            "Idéal pour travailler, le Wi-Fi est rapide et les sièges confortables.", # French: Ideal for working, the Wi-Fi is fast and the seats are comfortable.
            "La terrasse offre une vue magnifique sur la ville.", # French: The terrace offers a magnificent view of the city.
            "Le personnel est très accueillant et le service est rapide.", # French: The staff is very welcoming and the service is fast.
            "Un endroit calme pour lire un livre avec un bon café.", # French: A quiet place to read a book with good coffee.
            "Les options végétaliennes sont délicieuses et variées.", # French: The vegan options are delicious and varied.
            "Le parking est facile d'accès, ce qui est un plus.", # French: Parking is easy to access, which is a plus.
            "C'est un lieu artistique avec une décoration unique.", # French: It's an artistic place with unique decoration.
            "Parfait pour une sortie en famille, il y a de l'espace pour les enfants.", # French: Perfect for a family outing, there is space for children.
            "Les desserts sont incroyables, surtout le tiramisu.", # French: The desserts are incredible, especially the tiramisu.
            "Un endroit très animé, idéal pour faire la fête.", # French: A very lively place, ideal for partying.
            "L'accès en fauteuil roulant est très bien pensé.", # French: Wheelchair access is very well thought out.
            "Ce café accepte les animaux de compagnie, c'est génial !", # French: This cafe accepts pets, that's great!
            "Les portions sont généreuses et le rapport qualité-prix est excellent." # French: Portions are generous and the value for money is excellent.
        ]
    }

    # Convert raw text list to the expected format [{'text': 'review content'}]
    all_reviews = [{'text': r} for r in all_reviews_data['texts']]

    # --- Simulate various places with mixed language reviews to show preferences ---

    # Place 1: Cozy & Quiet Cafe (mix of English, Arabic, French)
    place_reviews_cozy_cafe = [
        all_reviews[0],   # English: "Absolutely fantastic! The best I've ever had." (Positive, general)
        all_reviews[2],   # Arabic: "المكان رايق والقاعدة بتاعته حلوة، أكيد هروح تاني." (Calm, nice seating -> Quiet, Cozy)
        all_reviews[16],  # Arabic: "مره عجبني المكان، رايق وهادي." (Calm and quiet -> Quiet, Cozy)
        all_reviews[39],  # Arabic: "مكان هادئ ومريح للأعصاب." (Quiet and relaxing -> Quiet, Cozy)
        all_reviews[97],  # French: "Un endroit calme pour lire un livre avec un bon café." (Quiet, Coffee, Cozy)
        all_reviews[95]   # French: "L'ambiance est décontractée, parfaite pour une soirée entre amis." (Casual)
    ]

    # Place 2: Trendy & Lively Restaurant (mix of English, Arabic, French)
    place_reviews_trendy_restaurant = [
        all_reviews[1],   # Arabic: "الأكل كان تحفة والخدمة ممتازة، مكان يستاهل التجربة." (Excellent food/service)
        all_reviews[5],   # English: "The food was good, but it was a bit too crowded." (Lively)
        all_reviews[22],  # English: "The ambiance is perfect for a date night." (Romantic, Date)
        all_reviews[25],  # Arabic: "المنظر من هناك تحفة." (Amazing view -> Scenic View)
        all_reviews[93],  # French: "Ce restaurant est très élégant et les plats sont gastronomiques." (Elegant, Gourmet)
        all_reviews[101]  # French: "Un endroit très animé, idéal pour faire la fête." (Lively)
    ]

    # Place 3: Family-Friendly & Accessible Spot (mix of English, Arabic, French)
    place_reviews_family_spot = [
        all_reviews[31],  # Arabic: "فريق العمل ودود للغاية ومتعاون." (Friendly staff)
        all_reviews[46],  # Arabic: "أنصح فيه للعائلات." (Recommend for families -> Family-Friendly)
        all_reviews[75],  # English: "The portions were generous and delicious." (Comfort Food)
        all_reviews[100], # French: "Parfait pour une sortie en famille, il y a de l'espace pour les enfants." (Family-Friendly)
        all_reviews[102]  # French: "L'accès en fauteuil roulant est très bien pensé." (Wheelchair Accessible)
    ]

    # Place 4: Workspace & Coffee Spot (mix of English, Arabic, Tunisian, French)
    place_reviews_workspace_cafe = [
        all_reviews[29],  # English: "The coffee here is top-notch." (Coffee)
        all_reviews[30],  # Arabic: "كل شيء كان نظيفاً ومنظماً." (Clean)
        all_reviews[94],  # French: "Idéal pour travailler, le Wi-Fi est rapide et les sièges confortables." (Workspace, Wi-Fi Available)
        all_reviews[44],  # Arabic: "القهوة عندهم مضبوطة." (Coffee is perfect)
        all_reviews[33],  # Arabic (Levantine): "اللقمات طيبين والمحل بجنن." (Good food, amazing place)
        all_reviews[8]    # Tunisian: "بنة على بنة، يعطيهم الصحة." (Delicious)
    ]

    # Place 5: Artistic & Bohemian Place (mix of English, Arabic, French)
    place_reviews_artistic_bohemian = [
        all_reviews[43],  # English: "A hidden gem in the city."
        all_reviews[40],  # Arabic: "الديكور بسيط وحلو." (Decor is simple and nice -> Casual, Artistic)
        all_reviews[64],  # Arabic: "الديكورات الداخلية مذهلة." (Interior decorations are amazing -> Artistic, Elegant)
        all_reviews[99]   # French: "C'est un lieu artistique avec une décoration unique." (Artistic, Bohemian)
    ]

    # Place with no reviews
    place_reviews_empty = []

    # Place with reviews that are empty strings or None
    place_reviews_only_empty_text = [{'text': ''}, {'text': '   '}, {'text': None}]


    print("\n--- Place Attribute Profiles (Multi-Language Reviews) ---")

    profile_cozy_cafe = calculate_place_attribute_profile(place_reviews_cozy_cafe, embedder, predefined_attributes)
    print(f"\nProfile for Cozy Cafe: {profile_cozy_cafe}")

    profile_trendy_restaurant = calculate_place_attribute_profile(place_reviews_trendy_restaurant, embedder, predefined_attributes)
    print(f"\nProfile for Trendy Restaurant: {profile_trendy_restaurant}")

    profile_family_spot = calculate_place_attribute_profile(place_reviews_family_spot, embedder, predefined_attributes)
    print(f"\nProfile for Family-Friendly Spot: {profile_family_spot}")

    profile_workspace_cafe = calculate_place_attribute_profile(place_reviews_workspace_cafe, embedder, predefined_attributes)
    print(f"\nProfile for Workspace Cafe: {profile_workspace_cafe}")

    profile_artistic_bohemian = calculate_place_attribute_profile(place_reviews_artistic_bohemian, embedder, predefined_attributes)
    print(f"\nProfile for Artistic/Bohemian Place: {profile_artistic_bohemian}")

    profile_empty = calculate_place_attribute_profile(place_reviews_empty, embedder, predefined_attributes)
    print(f"\nProfile for Empty Reviews Place: {profile_empty}")

    profile_only_empty_text = calculate_place_attribute_profile(place_reviews_only_empty_text, embedder, predefined_attributes)
    print(f"\nProfile for Only Empty Text Reviews Place: {profile_only_empty_text}")

    # --- Preference Scores with Multi-Language Profiles ---
    print("\n--- Preference Scores (Multi-Language Examples) ---")

    # User 1: Prefers Cozy and Quiet
    user_prefs_1 = {"Cozy": 0.6, "Quiet": 0.4}
    score_cozy_cafe_user_1 = calculate_preference_score(user_prefs_1, profile_cozy_cafe, predefined_attributes)
    score_trendy_restaurant_user_1 = calculate_preference_score(user_prefs_1, profile_trendy_restaurant, predefined_attributes)
    print(f"\nUser 1 (Cozy/Quiet) - Score for Cozy Cafe: {score_cozy_cafe_user_1:.4f}")
    print(f"User 1 (Cozy/Quiet) - Score for Trendy Restaurant: {score_trendy_restaurant_user_1:.4f}")

    # User 2: Prefers Trendy and Lively
    user_prefs_2 = {"Trendy": 0.5, "Lively": 0.5}
    score_cozy_cafe_user_2 = calculate_preference_score(user_prefs_2, profile_cozy_cafe, predefined_attributes)
    score_trendy_restaurant_user_2 = calculate_preference_score(user_prefs_2, profile_trendy_restaurant, predefined_attributes)
    print(f"\nUser 2 (Trendy/Lively) - Score for Cozy Cafe: {score_cozy_cafe_user_2:.4f}")
    print(f"User 2 (Trendy/Lively) - Score for Trendy Restaurant: {score_trendy_restaurant_user_2:.4f}")

    # User 3: Prefers Family-Friendly
    user_prefs_3 = {"Family-Friendly": 1.0}
    score_cozy_cafe_user_3 = calculate_preference_score(user_prefs_3, profile_cozy_cafe, predefined_attributes)
    score_family_spot_user_3 = calculate_preference_score(user_prefs_3, profile_family_spot, predefined_attributes)
    print(f"\nUser 3 (Family-Friendly) - Score for Cozy Cafe: {score_cozy_cafe_user_3:.4f}")
    print(f"User 3 (Family-Friendly) - Score for Family-Friendly Spot: {score_family_spot_user_3:.4f}")

    # User 4: Prefers Workspace and Wi-Fi
    user_prefs_4 = {"Workspace": 0.7, "Wi-Fi Available": 0.3}
    score_cozy_cafe_user_4 = calculate_preference_score(user_prefs_4, profile_cozy_cafe, predefined_attributes)
    score_workspace_cafe_user_4 = calculate_preference_score(user_prefs_4, profile_workspace_cafe, predefined_attributes)
    print(f"\nUser 4 (Workspace/Wi-Fi) - Score for Cozy Cafe: {score_cozy_cafe_user_4:.4f}")
    print(f"User 4 (Workspace/Wi-Fi) - Score for Workspace Cafe: {score_workspace_cafe_user_4:.4f}")

    # User 5: Prefers Gourmet and Elegant
    user_prefs_5 = {"Gourmet": 0.5, "Elegant": 0.5}
    score_trendy_restaurant_user_5 = calculate_preference_score(user_prefs_5, profile_trendy_restaurant, predefined_attributes)
    print(f"\nUser 5 (Gourmet/Elegant) - Score for Trendy Restaurant: {score_trendy_restaurant_user_5:.4f}")

    # User 6: Prefers Artistic and Bohemian
    user_prefs_6 = {"artistic": 0.6, "Bohemian": 0.4}
    score_artistic_bohemian_user_6 = calculate_preference_score(user_prefs_6, profile_artistic_bohemian, predefined_attributes)
    print(f"\nUser 6 (Artistic/Bohemian) - Score for Artistic/Bohemian Place: {score_artistic_bohemian_user_6:.4f}")

    # Example User 7: No preferences specified (should return 0)
    user_prefs_7 = {}
    score_cozy_cafe_user_7 = calculate_preference_score(user_prefs_7, profile_cozy_cafe, predefined_attributes)
    print(f"\nUser 7 (No Preferences) - Score for Cozy Cafe: {score_cozy_cafe_user_7:.4f}")
