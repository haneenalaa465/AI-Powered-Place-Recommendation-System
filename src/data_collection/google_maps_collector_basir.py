import pandas as pd
import asyncio
import os
import sys
from datetime import datetime
import googlemaps
from googlemaps.exceptions import ApiError as GoogleMapsApiError
import nest_asyncio
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

nest_asyncio.apply()

# --- Data Models  ---
@dataclass
class Place:
    # Basir dataset fields:
    name: str
    country: str 
    city: str 
    cuisine: str
    recommended_dish: str
    avg_price_usd: float
    budget_range: str
    suitability: str
    
    # google maps API fields
    address: str
    rating: float
    num_reviews: int
    price_level: str
    latitude: float
    longitude: float
    place_id: str
    link: str

@dataclass
class Review:
    place_name: str
    review_index: int
    author: str
    text: str
    rating: float
    language: str

class GoogleMapsCollectorBasir:
    def __init__(self, api_key: str, input_csv_path: str, output_dir: str):
        """
        Init. the Google Maps Collector for Basir dataset.

        Args:
            api_key (str):  Google Maps API key.
            input_csv_path (str): Path to the Basir Research dataset CSV.
            output_dir (str): a dir to save raw collected data
        """
        if not api_key:
            raise ValueError("Google Maps API Key is required.")
        self.gmaps = googlemaps.Client(key=api_key)
        self.input_csv_path = input_csv_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"GoogleMapsCollector initialized. Output directory: {self.output_dir}")

    def _geocode_place(self, address: str):
        """
        Geocodes an address to get its latitude, longitude, and Google Place ID.
        """
        try:
            geocode_result = self.gmaps.geocode(address)
            if geocode_result:
                location = geocode_result[0]["geometry"]["location"]
                place_id = geocode_result[0].get("place_id", "N/A")
                return location["lat"], location["lng"], place_id
            print(f"no geocoding result found for '{address}'.")
            return 0.0, 0.0, "N/A"
        except GoogleMapsApiError as e:
            print(f"error during geocoding for '{address}': {e}")
            return 0.0, 0.0, "N/A"
        except Exception as e:
            print(f"unexpected error '{address}': {e}")
            return 0.0, 0.0, "N/A"

    def _get_place_details_and_reviews(self, place_id: str):
        """
        Fetches detailed information for a Google Place ID, including overall rating,
        total number of reviews, price level, official URL of the location, and a limited set of ~5 reviews.
        """
        if place_id == "N/A" or place_id is None:
            print(f"null place_id, skip its details ")
            return {"rating": 0.0, "user_ratings_total": 0, "price_level": "unknown", "url": "N/A", "reviews": []}
        try:
            result = self.gmaps.place(place_id=place_id, fields=["rating", "user_ratings_total", "price_level", "url", "review"]).get("result", {})
            return {
                "rating": result.get("rating", 0.0),
                "user_ratings_total": result.get("user_ratings_total", 0),
                "price_level": result.get("price_level", "unknown"),
                "url": result.get("url", "N/A"),
                "reviews": result.get("reviews", []) # list of review dicts
            }
        except GoogleMapsApiError as e:
            print(f"error fetching details for place ID {place_id}: {e}")
            return {"rating": 0.0, "user_ratings_total": 0, "price_level": "unknown", "url": "N/A", "reviews": []}
        except Exception as e:
            print(f"ERROR: Unexpected error fetching details for place ID {place_id}: {e}")
            return {"rating": 0.0, "user_ratings_total": 0, "price_level": "unknown", "url": "N/A", "reviews": []}

    async def collect_data(self, max_places: Optional[int] = None, save_interval: int = 50):
        """
        Handles the fetching process: geocodes restaurants, fetches details and reviews,
        and saves data incrementally with checkpoints.
        Args:
            max_places (int, optional): Maximum number of places to get for
            save_interval (int): How often to save partial data for checkpointing.
        """
        df = pd.read_csv(self.input_csv_path)
        
        if max_places is not None:
            df = df.head(max_places)

        total_restaurants = len(df)
        places_list: List[Place] = [] #  stores the objs of Place 
        all_raw_reviews: List[Review] = [] # stores the objs of Review 
        failed_addresses: List[str] = [] # addresses that failed geocoding

        for idx, row in df.iterrows():
            restaurant_name = row.get("Restaurant Name", "Unknown Name")
            original_city = row.get("City", "Unknown City")
            original_country = row.get("Country", "Unknown Country")
            address_query = f"{restaurant_name}, {original_city}, {original_country}"
            
            # geocode the address to get lat, lon + place_id
            lat, lon, place_id = self._geocode_place(address_query)
            
            if place_id == "N/A" or place_id is None:
                print(f"invalid place_id for '{restaurant_name}' (Index: {idx})")
                failed_addresses.append(address_query)
                places_list.append(Place(
                    name=restaurant_name, 
                    country=original_country, 
                    city=original_city,    
                    cuisine=row.get("Type of Cuisine", "N/A"),
                    recommended_dish=row.get("Recommended Dish", "N/A"),
                    avg_price_usd=row.get("Avg Price per Person (USD)", 0.0),
                    budget_range=row.get("Budget Range", "N/A"),
                    suitability=row.get("Suitability", "N/A"),
                    address=address_query, 
                    rating=0.0, num_reviews=0, price_level="N/A", latitude=0.0, longitude=0.0,
                    place_id="N/A", link="N/A"
                ))
                await asyncio.sleep(0.5) 
                continue 

            # fetching the detailes + reviews using the place_id
            details = self._get_place_details_and_reviews(place_id)

            place_obj = Place(
                name=restaurant_name,
                country=original_country,
                city=original_city,
                cuisine=row.get("Type of Cuisine", "N/A"),
                recommended_dish=row.get("Recommended Dish", "N/A"),
                avg_price_usd=row.get("Avg Price per Person (USD)", 0.0),
                budget_range=row.get("Budget Range", "N/A"),
                suitability=row.get("Suitability", "N/A"),
                address=details.get("formatted_address", address_query), 
                rating=details["rating"],
                num_reviews=details["user_ratings_total"],
                price_level=details["price_level"],
                latitude=lat,
                longitude=lon,
                place_id=place_id,
                link=details["url"]
            )
            places_list.append(place_obj)

            for i, review_data in enumerate(details["reviews"]): 
                all_raw_reviews.append(
                    Review(
                        place_name=place_obj.name, 
                        review_index=i+1,
                        author=review_data.get("author_name", "N/A"),
                        text=review_data.get("text", ""),
                        rating=review_data.get("rating", 0.0),
                        language=review_data.get("language", "unknown")
                    )
                )
            
            await asyncio.sleep(0.5) 

            # saves partial data for checkpointing
            if len(places_list) % save_interval == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                partial_places_df = pd.DataFrame([asdict(p) for p in places_list])
                partial_reviews_df = pd.DataFrame([asdict(r) for r in all_raw_reviews])
                
                # save to CSV files
                partial_places_df.to_csv(f"{self.output_dir}/checkpoint_places_{len(places_list)}_{timestamp}.csv", index=False)
                partial_reviews_df.to_csv(f"{self.output_dir}/checkpoint_reviews_{len(all_raw_reviews)}_{timestamp}.csv", index=False)
                print(f"[CHECKPOINT] Saved partial data: {len(places_list)} places, {len(all_raw_reviews)} reviews.")

        # --- Final Data Saving ---
        final_places_df = pd.DataFrame([asdict(p) for p in places_list])
        final_reviews_df = pd.DataFrame([asdict(r) for r in all_raw_reviews])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_places_df.to_csv(f"{self.output_dir}/scraped_places_final_{timestamp}.csv", index=False)
        final_reviews_df.to_csv(f"{self.output_dir}/scraped_reviews_final_{timestamp}.csv", index=False)

        # places that failed geocoding
        with open(f"{self.output_dir}/failed_geocoding_final_{timestamp}.txt", "w") as f:
            for addr in failed_addresses:
                f.write(addr + "\n")

        print(f"\nFinal data saved:")
        print(f"  {len(final_places_df)} places to {self.output_dir}/scraped_places_final_{timestamp}.csv")
        print(f"  {len(final_reviews_df)} reviews to {self.output_dir}/scraped_reviews_final_{timestamp}.csv")
        print(f"  {len(failed_addresses)} failed geocoding attempts logged to {self.output_dir}/failed_geocoding_final_{timestamp}.txt")
        
        return final_places_df, final_reviews_df
    



    # --- Main Execution  ---
if __name__ == "__main__":
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_file_dir)) 
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from config.settings import MAPS_API_KEY, RAW_DATA_DIR, BASIR_INPUT_CSV_PATH 
    
    maps_api_key = MAPS_API_KEY
    output_data_dir = RAW_DATA_DIR
    input_csv_file_path = BASIR_INPUT_CSV_PATH
        
    collector = GoogleMapsCollectorBasir(
        api_key=maps_api_key,
        input_csv_path=input_csv_file_path,
        output_dir=output_data_dir
    )

    places_df, reviews_df = asyncio.run(collector.collect_data(max_places=5)) 
    print(" collection completed")

    print("\n=== Restaurants ===")
    print(places_df.head()[["name", "city", "country", "address", "cuisine", "rating", "latitude", "longitude"]])

    print("\n=== Reviews ===")
    print(reviews_df.head()[["place_name", "text", "rating"]])