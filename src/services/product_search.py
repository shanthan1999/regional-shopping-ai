"""
Product search integration with popular Indian grocery platforms
"""

import requests
import json
import time
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Optional
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import logging

# DuckDuckGo Shopping Service will be imported where needed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductSearchService:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def search_all_platforms(self, query: str, limit: int = 5) -> Dict[str, List[Dict]]:
        """Search across multiple platforms and return aggregated results"""
        results = {
            'zepto': [],
            'blinkit': [],
            'bigbasket': [],
            'amazon_fresh': [],
            'duckduckgo_shopping': []
        }
        
        # Use ThreadPoolExecutor for parallel searches
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                'zepto': executor.submit(self.search_zepto, query, limit),
                'blinkit': executor.submit(self.search_blinkit, query, limit),
                'bigbasket': executor.submit(self.search_bigbasket, query, limit),
            }
            
            # Collect results as they complete
            for platform, future in futures.items():
                try:
                    results[platform] = future.result(timeout=10)
                except Exception as e:
                    logger.error(f"Error searching {platform}: {e}")
                    results[platform] = []
        
        # Add DuckDuckGo Shopping search
        try:
            from src.services.duckduckgo_shopping_search import duckduckgo_shopping_service
            results['duckduckgo_shopping'] = duckduckgo_shopping_service.search_products(query, limit)
        except Exception as e:
            logger.error(f"Error searching DuckDuckGo Shopping: {e}")
            results['duckduckgo_shopping'] = []
            
        return results
    
    def search_zepto(self, query: str, limit: int = 5) -> List[Dict]:
        """Search Zepto for products"""
        try:
            # Zepto API endpoint (this is a mock - real endpoint would need API key)
            # In practice, you'd need to reverse engineer their API or use official API
            search_url = f"https://user-app-api.zeptonow.com/api/v3/search"
            
            params = {
                'query': query,
                'page_size': limit,
                'store_id': 'default'  # This would be location-specific
            }
            
            # Mock response for demonstration
            # mock_products = self._get_mock_zepto_products(query, limit)
            # return mock_products
            return [] # Placeholder for real Zepto API integration
            
        except Exception as e:
            logger.error(f"Zepto search error: {e}")
            return []
    
    def search_blinkit(self, query: str, limit: int = 5) -> List[Dict]:
        """Search Blinkit for products"""
        try:
            # Blinkit search (mock implementation)
            # mock_products = self._get_mock_blinkit_products(query, limit)
            # return mock_products
            return [] # Placeholder for real Blinkit API integration
            
        except Exception as e:
            logger.error(f"Blinkit search error: {e}")
            return []
    
    def search_bigbasket(self, query: str, limit: int = 5) -> List[Dict]:
        """Search BigBasket for products"""
        try:
            # BigBasket search
            search_url = "https://www.bigbasket.com/product/get-products/"
            
            params = {
                'q': query,
                'n': limit
            }
            
            # Mock response for demonstration
            # mock_products = self._get_mock_bigbasket_products(query, limit)
            # return mock_products
            return [] # Placeholder for real BigBasket API integration
            
        except Exception as e:
            logger.error(f"BigBasket search error: {e}")
            return []
    
    def search_duckduckgo_shopping(self, query: str, limit: int = 5) -> List[Dict]:
        """Search DuckDuckGo Shopping for grocery products"""
        try:
            from src.services.duckduckgo_shopping_search import duckduckgo_shopping_service
            products = duckduckgo_shopping_service.search_products(query, limit)
            return products
            
        except Exception as e:
            logger.error(f"DuckDuckGo Shopping search error: {e}")
            return []

    def get_best_deals(self, query: str) -> Dict:
        """Get best deals across all platforms"""
        all_results = self.search_all_platforms(query, limit=3)
        
        # Aggregate and find best deals
        all_products = []
        for platform, products in all_results.items():
            all_products.extend(products)
        
        if not all_products:
            return {'best_deals': [], 'price_comparison': {}}
        
        # Sort by price (extract numeric value)
        def extract_price(price_str):
            try:
                return float(re.findall(r'[\d.]+', price_str.replace('₹', ''))[0])
            except:
                return float('inf')
        
        sorted_products = sorted(all_products, key=lambda x: extract_price(x['price']))
        
        # Price comparison by platform
        price_comparison = {}
        for platform, products in all_results.items():
            if products:
                avg_price = sum(extract_price(p['price']) for p in products) / len(products)
                price_comparison[platform] = {
                    'average_price': f'₹{avg_price:.0f}',
                    'product_count': len(products),
                    'fastest_delivery': min(products, key=lambda x: x.get('delivery_time', '999 mins'))['delivery_time']
                }
        
        return {
            'best_deals': sorted_products[:5],
            'price_comparison': price_comparison,
            'total_products_found': len(all_products)
        }

# Global instance
product_search_service = ProductSearchService()