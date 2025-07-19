"""
DuckDuckGo Shopping search integration using duckpy
"""

import json
from typing import List, Dict, Optional
import time
import re
from duckpy import Client # Import the duckpy Client

class DuckPyShoppingService:
    def __init__(self):
        self.client = Client() # Initialize duckpy Client

    def search_products(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search DuckDuckGo for products using duckpy's general search.
        """
        try:
            search_query = f"{query} shopping" 
            results = self.client.search(search_query)
            
            products = []
            for item in results:
                product_name = item.title if hasattr(item, 'title') else 'N/A'
                product_url = item.url if hasattr(item, 'url') else 'N/A'
                description = item.description if hasattr(item, 'description') else 'N/A'

                # Validate product_url
                if not (isinstance(product_url, str) and product_url.startswith(('http://', 'https://'))):
                    continue  # Skip products with invalid URLs

                price_match = None
                price_re = r'₹(\d[\d,]*\.?\d{0,2})|$(\d[\d,]*\.?\d{0,2})'
                
                for text_to_search in [product_name, description]:
                    if text_to_search != 'N/A':
                        match = re.search(price_re, text_to_search)
                        if match:
                            price_match = match.group(0)
                            break

                products.append({
                    'name': product_name,
                    'price': price_match if price_match else 'N/A',
                    'original_price': 'N/A',
                    'discount': 'N/A',
                    'seller': 'DuckDuckGo Search',
                    'rating': 'N/A',
                    'reviews': 'N/A',
                    'image_url': 'https://via.placeholder.com/150?text=DuckDuckGo+Product',
                    'product_url': product_url,
                    'availability': 'Check link',
                    'delivery': 'N/A',
                    'category': 'General',
                    'source': 'duckpy_search',
                    'description': description
                })
            
            return products[:limit]

        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []

    def get_price_comparison(self, query: str) -> Dict:
        """
        Get price comparison for a product from DuckDuckGo.
        Not directly supported by duckpy's general search.
        """
        return {
            'query': query,
            'recommendations': {'cheapest': None, 'fastest_delivery': None},
            'price_trends': {},
            'comparison_data': [],
            'error': 'DuckDuckGo does not provide structured price comparison via general search.'
        }

    def get_trending_products(self, category: str = None) -> List[Dict]:
        """
        Get trending products from DuckDuckGo.
        Not directly supported by duckpy's general search.
        """
        return []

# Global instance
duckpy_shopping_service = DuckPyShoppingService() 