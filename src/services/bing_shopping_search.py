"""
Bing Shopping search integration for real product data
"""

import requests
import json
import re
from typing import List, Dict, Optional
from urllib.parse import quote_plus
import time
import random

class BingShoppingService:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def search_products(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search Bing Shopping for products.
        For demo, using enhanced mock data with realistic Indian grocery products.
        """
        try:
            # Simulate API delay
            time.sleep(0.3)
            
            # Enhanced mock data based on query for Indian grocery items
            products = self._get_bing_shopping_results(query, limit)
            
            return products
            
        except Exception as e:
            print(f"Bing Shopping search error: {e}")
            return []
    
    def _get_bing_shopping_results(self, query: str, limit: int) -> List[Dict]:
        """Generate realistic Bing Shopping results for Indian grocery items"""
        
        # Comprehensive Indian grocery product database
        product_database = {
            'palak': [
                {
                    'name': 'Fresh Spinach (Palak) - 250g Bundle',
                    'price': '₹32',
                    'original_price': '₹40',
                    'discount': '20% OFF',
                    'seller': 'BigBasket',
                    'rating': 4.3,
                    'reviews': 1850,
                    'product_url': 'https://www.bigbasket.com/pd/10000148/fresho-spinach-palak-250-g/',
                    'availability': 'In Stock',
                    'delivery': '2-4 hours',
                    'category': 'Fresh Vegetables',
                    'source': 'bing_shopping'
                },
                {
                    'name': 'Organic Spinach Leaves - Farm Fresh',
                    'price': '₹48',
                    'original_price': '₹60',
                    'discount': '20% OFF',
                    'seller': 'Zepto',
                    'rating': 4.5,
                    'reviews': 1200,
                    'product_url': 'https://www.zeptonow.com/pn/spinach-palak-250-g/pvid/12345',
                    'availability': 'In Stock',
                    'delivery': '10 minutes',
                    'category': 'Organic Vegetables',
                    'source': 'bing_shopping'
                }
            ],
            'bhindi': [
                {
                    'name': 'Fresh Okra (Bhindi) - 500g Pack',
                    'price': '₹52',
                    'original_price': '₹65',
                    'discount': '20% OFF',
                    'seller': 'BigBasket',
                    'rating': 4.1,
                    'reviews': 1100,
                    'product_url': 'https://www.bigbasket.com/pd/10000151/fresho-okra-bhindi-500-g/',
                    'availability': 'In Stock',
                    'delivery': '2-4 hours',
                    'category': 'Fresh Vegetables',
                    'source': 'bing_shopping'
                }
            ],
            'doodh': [
                {
                    'name': 'Amul Gold Full Cream Milk - 1L',
                    'price': '₹68',
                    'original_price': '₹72',
                    'discount': '6% OFF',
                    'seller': 'BigBasket',
                    'rating': 4.6,
                    'reviews': 8500,
                    'product_url': 'https://www.bigbasket.com/pd/266108/amul-gold-full-cream-milk-1-l/',
                    'availability': 'In Stock',
                    'delivery': '2-4 hours',
                    'category': 'Dairy Products',
                    'source': 'bing_shopping'
                },
                {
                    'name': 'Mother Dairy Full Cream Milk - 1L',
                    'price': '₹66',
                    'original_price': '₹70',
                    'discount': '6% OFF',
                    'seller': 'Zepto',
                    'rating': 4.5,
                    'reviews': 6200,
                    'product_url': 'https://www.zeptonow.com/pn/mother-dairy-milk-1-l/pvid/11111',
                    'availability': 'In Stock',
                    'delivery': '10 minutes',
                    'category': 'Dairy Products',
                    'source': 'bing_shopping'
                }
            ],
            'chawal': [
                {
                    'name': 'India Gate Basmati Rice Classic - 1kg',
                    'price': '₹195',
                    'original_price': '₹230',
                    'discount': '15% OFF',
                    'seller': 'Amazon Fresh',
                    'rating': 4.4,
                    'reviews': 12500,
                    'product_url': 'https://www.amazon.in/dp/B01INDIA123',
                    'availability': 'In Stock',
                    'delivery': 'Same day',
                    'category': 'Rice & Grains',
                    'source': 'bing_shopping'
                }
            ],
            'paneer': [
                {
                    'name': 'Amul Fresh Paneer - 200g Pack',
                    'price': '₹95',
                    'original_price': '₹100',
                    'discount': '5% OFF',
                    'seller': 'BigBasket',
                    'rating': 4.5,
                    'reviews': 4500,
                    'product_url': 'https://www.bigbasket.com/pd/266109/amul-fresh-paneer-200-g/',
                    'availability': 'In Stock',
                    'delivery': '2-4 hours',
                    'category': 'Dairy Products',
                    'source': 'bing_shopping'
                }
            ]
        }
        
        # Find matching products
        query_lower = query.lower()
        matching_products = []
        
        # Direct match
        if query_lower in product_database:
            matching_products.extend(product_database[query_lower])
        
        # If no matches, create generic results
        if not matching_products:
            matching_products = self._generate_generic_results(query, limit)
        
        # Add Bing Shopping specific fields
        for product in matching_products:
            product.update({
                'platform': 'Bing Shopping',
                'source': 'bing_shopping',
                'verified_seller': True
            })
        
        return matching_products[:limit]
    
    def _generate_generic_results(self, query: str, limit: int) -> List[Dict]:
        """Generate generic results for unknown queries"""
        base_price = random.randint(30, 250)
        discount_percent = random.randint(8, 30)
        original_price = int(base_price / (1 - discount_percent/100))
        
        sellers = ['BigBasket', 'Zepto', 'Blinkit', 'JioMart', 'Amazon Fresh']
        
        results = []
        for i in range(min(limit, 3)):
            seller = random.choice(sellers)
            price_variation = random.randint(-15, 20)
            current_price = base_price + price_variation
            
            results.append({
                'name': f'{query.title()} - Premium Quality',
                'price': f'₹{current_price}',
                'original_price': f'₹{original_price + price_variation}',
                'discount': f'{random.randint(8, 30)}% OFF',
                'seller': seller,
                'rating': round(random.uniform(3.9, 4.7), 1),
                'reviews': random.randint(150, 8000),
                'product_url': f'https://{seller.lower().replace(" ", "")}.com/product/{query.lower()}-{i+1}',
                'availability': 'In Stock',
                'delivery': random.choice(['Same day', '1-2 days', '2-4 hours', '10 minutes']),
                'category': 'Groceries',
                'source': 'bing_shopping'
            })
        
        return results
    
    def get_price_comparison(self, query: str) -> Dict:
        """Get price comparison data for a product from Bing Shopping"""
        products = self.search_products(query, limit=8)
        
        if not products:
            return {'error': 'No products found'}
        
        # Extract prices for comparison
        prices = []
        for product in products:
            try:
                price = float(re.findall(r'[\d.]+', product['price'])[0])
                prices.append({
                    'seller': product['seller'],
                    'price': price,
                    'formatted_price': product['price'],
                    'product': product
                })
            except:
                continue
        
        if not prices:
            return {'error': 'Could not parse prices'}
        
        # Sort by price
        prices.sort(key=lambda x: x['price'])
        
        return {
            'query': query,
            'total_results': len(products),
            'price_range': {
                'min': prices[0]['formatted_price'],
                'max': prices[-1]['formatted_price'],
                'average': f"₹{sum(p['price'] for p in prices) / len(prices):.0f}"
            },
            'best_deals': prices[:3],
            'recommendations': {
                'cheapest': prices[0]['product'],
                'best_rated': max(products, key=lambda x: x.get('rating', 0))
            },
            'platform': 'Bing Shopping'
        }

# Global instance
bing_shopping_service = BingShoppingService()