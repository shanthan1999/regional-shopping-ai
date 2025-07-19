from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import json
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from difflib import SequenceMatcher
import requests
from langdetect import detect
import hashlib
import time
import urllib.parse

rag_bp = Blueprint('rag', __name__)

# Global variables to store models and data
embedding_model = None
vector_index = None
knowledge_base = None
llm_pipeline = None
embedding_to_entry_map = None
translation_cache = {}  # In-memory cache for translations

def initialize_models():
    """Initialize the embedding model, vector index, and LLM pipeline."""
    global embedding_model, vector_index, knowledge_base, llm_pipeline
    
    print("Initializing models...")
    
    # Initialize embedding model
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # Initialize LLM pipeline (using a smaller model for demo purposes)
    # In production, you would use Llama3-8B or similar
    llm_pipeline = pipeline(
        "text-generation",
        model="microsoft/DialoGPT-medium",
        tokenizer="microsoft/DialoGPT-medium",
        device=-1  # Use CPU
    )
    
    # Load or create knowledge base and vector index
    load_or_create_knowledge_base()
    
    # Load translation cache
    load_translation_cache()
    
    print("Models initialized successfully!")

def load_translation_cache():
    """Load cached translations from disk."""
    global translation_cache
    
    cache_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'translation_cache.pkl')
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                translation_cache = pickle.load(f)
            print(f"Loaded {len(translation_cache)} cached translations")
        except Exception as e:
            print(f"Error loading translation cache: {e}")
            translation_cache = {}
    else:
        translation_cache = {}

def save_translation_cache():
    """Save cached translations to disk."""
    cache_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'translation_cache.pkl')
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(translation_cache, f)
    except Exception as e:
        print(f"Error saving translation cache: {e}")

def get_cache_key(word, source_lang='hi', target_lang='en'):
    """Generate a cache key for translation."""
    return hashlib.md5(f"{word.lower()}_{source_lang}_{target_lang}".encode()).hexdigest()

def translate_word_simple(word, source_lang='hi', target_lang='en'):
    """Simple translation using Google Translate API via requests."""
    global translation_cache
    
    cache_key = get_cache_key(word, source_lang, target_lang)
    
    # Check cache first
    if cache_key in translation_cache:
        return translation_cache[cache_key]
    
    # Use a comprehensive dictionary-based approach for common words
    simple_translations = {
        # Vegetables
        'palak': 'spinach',
        'bhindi': 'okra',
        'karela': 'bitter gourd',
        'methi': 'fenugreek',
        'dhaniya': 'coriander',
        'pudina': 'mint',
        'adrak': 'ginger',
        'lehsun': 'garlic',
        'haldi': 'turmeric',
        'mirch': 'chili',
        'jeera': 'cumin',
        'gajar': 'carrot',
        'gobi': 'cauliflower',
        'lauki': 'bottle gourd',
        'baingan': 'eggplant',
        'aloo': 'potato',
        'tamatar': 'tomato',
        'pyaaz': 'onion',
        'kela': 'banana',
        'aam': 'mango',
        
        # Dairy and liquids
        'doodh': 'milk',
        'paal': 'milk',
        'paneer': 'cottage cheese',
        'dahi': 'yogurt',
        'ghee': 'clarified butter',
        'makhan': 'butter',
        'malai': 'cream',
        'lassi': 'buttermilk',
        'chaas': 'buttermilk',
        'paani': 'water',
        'jal': 'water',
        
        # Grains and legumes
        'chawal': 'rice',
        'dal': 'lentils',
        'daal': 'lentils',
        'roti': 'bread',
        'chapati': 'flatbread',
        'naan': 'bread',
        'atta': 'flour',
        'besan': 'gram flour',
        'suji': 'semolina',
        
        # Spices and herbs
        'masala': 'spice mix',
        'garam': 'hot',
        'namak': 'salt',
        'cheeni': 'sugar',
        'gud': 'jaggery',
        'shahad': 'honey',
        'tel': 'oil',
        'sarson': 'mustard',
        'til': 'sesame',
        'kothimbir': 'coriander',
        
        # Food items
        'sabzi': 'vegetable curry',
        'curry': 'curry',
        'biryani': 'rice dish',
        'pulao': 'rice dish',
        'raita': 'yogurt side dish',
        'chutney': 'sauce',
        'pickle': 'pickle',
        'achar': 'pickle',
        'papad': 'crispy bread',
        'samosa': 'fried pastry',
        'pakora': 'fritters',
        'kachori': 'stuffed pastry',
        'paratha': 'stuffed bread',
        
        # Common words
        'khana': 'food',
        'bhojan': 'meal',
        'nashta': 'breakfast',
        'dopahar': 'lunch',
        'raat': 'dinner',
        'paisa': 'money',
        'rupaya': 'rupee',
        'ghar': 'home',
        'makaan': 'house',
        'bazar': 'market',
        'dukan': 'shop',
        'kitab': 'book',
        'kalam': 'pen',
        'kagaz': 'paper',
        
        # Body parts
        'aankhein': 'eyes',
        'haath': 'hands',
        'pair': 'feet',
        'sar': 'head',
        'baal': 'hair',
        'muh': 'mouth',
        'naak': 'nose',
        'kaan': 'ears',
        
        # Colors
        'safed': 'white',
        'kaala': 'black',
        'laal': 'red',
        'peela': 'yellow',
        'neela': 'blue',
        'hara': 'green',
        'gulabi': 'pink',
        'bhura': 'brown',
        
        # Common adjectives
        'accha': 'good',
        'bura': 'bad',
        'bada': 'big',
        'chota': 'small',
        'lamba': 'tall',
        'chouda': 'wide',
        'patla': 'thin',
        'mota': 'thick',
        'thand': 'cold',
        'garam': 'hot',
        'meetha': 'sweet',
        'namkeen': 'salty',
        'kadva': 'bitter',
        'khatta': 'sour'
    }
    
    word_lower = word.lower()
    if word_lower in simple_translations:
        translated_word = simple_translations[word_lower]
        translation_cache[cache_key] = translated_word
        
        # Save cache periodically
        if len(translation_cache) % 10 == 0:
            save_translation_cache()
        
        return translated_word
    
    # If not in simple dictionary, try Google Translate API
    try:
        # Use Google Translate API via requests
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            'client': 'gtx',
            'sl': source_lang,
            'tl': target_lang,
            'dt': 't',
            'q': word
        }
        
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result and len(result) > 0 and len(result[0]) > 0:
                translated_word = result[0][0][0].lower()
                
                # Only use translation if it's different from original
                if translated_word != word_lower:
                    # Cache the result
                    translation_cache[cache_key] = translated_word
                    
                    # Save cache periodically
                    if len(translation_cache) % 10 == 0:
                        save_translation_cache()
                    
                    return translated_word
    except Exception as e:
        print(f"Translation API error for '{word}': {e}")
    
    # Return original word if translation fails
    return word

def detect_language(text):
    """Detect the language of the text using multiple approaches."""
    try:
        # First, check if any word is definitely Hindi
        words = text.split()
        hindi_word_count = sum(1 for word in words if is_likely_hindi_word(word))
        total_words = len(words)
        
        if hindi_word_count > 0:
            if hindi_word_count == total_words:
                return 'hi'  # All words are Hindi
            else:
                return 'mixed'  # Mixed language
        
        # If no Hindi words detected, try langdetect
        detected_lang = detect(text)
        return detected_lang
    except:
        # If langdetect fails, do a simple heuristic
        words = text.split()
        if any(is_likely_hindi_word(word) for word in words):
            return 'hi'
        return 'en'  # Default to English if detection fails

def add_to_knowledge_base(hindi_word, english_word, category="user_added"):
    """Add a new translation to the knowledge base and vector index."""
    global knowledge_base, vector_index, embedding_model, embedding_to_entry_map
    
    # Create new entry
    new_entry = {
        "hindi_transliteration": hindi_word,
        "english_translation": english_word,
        "category": category,
        "variations": [hindi_word]
    }
    
    # Add to knowledge base
    knowledge_base.append(new_entry)
    entry_idx = len(knowledge_base) - 1
    
    # Create embeddings for the new entry
    texts_to_embed = [hindi_word, english_word]
    embeddings = embedding_model.encode(texts_to_embed)
    
    # Normalize embeddings
    faiss.normalize_L2(embeddings)
    
    # Add to vector index
    vector_index.add(embeddings.astype('float32'))
    
    # Update mapping
    for _ in texts_to_embed:
        embedding_to_entry_map.append(entry_idx)
    
    # Save updated knowledge base and index
    save_knowledge_base()
    
    print(f"Added new translation: {hindi_word} -> {english_word}")

def save_knowledge_base():
    """Save the updated knowledge base, vector index, and mapping to disk."""
    kb_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge_base.pkl')
    index_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector_index.faiss')
    map_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'embedding_map.pkl')
    
    try:
        with open(kb_path, 'wb') as f:
            pickle.dump(knowledge_base, f)
        faiss.write_index(vector_index, index_path)
        with open(map_path, 'wb') as f:
            pickle.dump(embedding_to_entry_map, f)
    except Exception as e:
        print(f"Error saving knowledge base: {e}")

def load_or_create_knowledge_base():
    """Load existing knowledge base or create a new one."""
    global vector_index, knowledge_base, embedding_to_entry_map
    
    kb_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge_base.pkl')
    index_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector_index.faiss')
    map_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'embedding_map.pkl')
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(kb_path), exist_ok=True)
    
    if os.path.exists(kb_path) and os.path.exists(index_path) and os.path.exists(map_path):
        # Load existing knowledge base
        with open(kb_path, 'rb') as f:
            knowledge_base = pickle.load(f)
        vector_index = faiss.read_index(index_path)
        with open(map_path, 'rb') as f:
            embedding_to_entry_map = pickle.load(f)
    else:
        # Create new knowledge base with sample data
        create_sample_knowledge_base()

def create_sample_knowledge_base():
    """Create a sample knowledge base with Hindi-English translations."""
    global vector_index, knowledge_base, embedding_model, embedding_to_entry_map
    
    # Expanded knowledge base entries
    knowledge_base = [
        {
            "hindi_transliteration": "kothimbir",
            "english_translation": "coriander",
            "category": "herbs",
            "variations": ["kothimbeer", "kotimbir", "kothimber"]
        },
        {
            "hindi_transliteration": "paal",
            "english_translation": "milk",
            "category": "dairy",
            "variations": ["pal", "pall"]
        },
        {
            "hindi_transliteration": "doodh",
            "english_translation": "milk",
            "category": "dairy",
            "variations": ["dudh", "dhood"]
        },
        {
            "hindi_transliteration": "chawal",
            "english_translation": "rice",
            "category": "grains",
            "variations": ["chaawal", "chaval"]
        },
        {
            "hindi_transliteration": "daal",
            "english_translation": "lentils",
            "category": "legumes",
            "variations": ["dal", "dhal"]
        },
        {
            "hindi_transliteration": "sabzi",
            "english_translation": "vegetables",
            "category": "produce",
            "variations": ["sabji", "subzi"]
        },
        {
            "hindi_transliteration": "aam",
            "english_translation": "mango",
            "category": "fruits",
            "variations": ["am", "aam"]
        },
        {
            "hindi_transliteration": "kela",
            "english_translation": "banana",
            "category": "fruits",
            "variations": ["kele", "kela"]
        },
        {
            "hindi_transliteration": "pyaaz",
            "english_translation": "onion",
            "category": "vegetables",
            "variations": ["pyaj", "piaz"]
        },
        {
            "hindi_transliteration": "aloo",
            "english_translation": "potato",
            "category": "vegetables",
            "variations": ["alu", "aaloo"]
        },
        {
            "hindi_transliteration": "tamatar",
            "english_translation": "tomato",
            "category": "vegetables",
            "variations": ["tamater", "tamaatar"]
        },
        {
            "hindi_transliteration": "palak",
            "english_translation": "spinach",
            "category": "vegetables",
            "variations": ["palak", "palakh"]
        },
        {
            "hindi_transliteration": "gajar",
            "english_translation": "carrot",
            "category": "vegetables",
            "variations": ["gaajar", "gajar"]
        },
        {
            "hindi_transliteration": "gobi",
            "english_translation": "cauliflower",
            "category": "vegetables",
            "variations": ["gobi", "gobhi"]
        },
        {
            "hindi_transliteration": "bhindi",
            "english_translation": "okra",
            "category": "vegetables",
            "variations": ["bhindi", "bindi"]
        },
        {
            "hindi_transliteration": "karela",
            "english_translation": "bitter gourd",
            "category": "vegetables",
            "variations": ["karela", "karella"]
        },
        {
            "hindi_transliteration": "lauki",
            "english_translation": "bottle gourd",
            "category": "vegetables",
            "variations": ["lauki", "loki"]
        },
        {
            "hindi_transliteration": "baingan",
            "english_translation": "eggplant",
            "category": "vegetables",
            "variations": ["baingan", "bengan"]
        },
        {
            "hindi_transliteration": "methi",
            "english_translation": "fenugreek",
            "category": "herbs",
            "variations": ["methi", "methhi"]
        },
        {
            "hindi_transliteration": "pudina",
            "english_translation": "mint",
            "category": "herbs",
            "variations": ["pudina", "pudeena"]
        },
        {
            "hindi_transliteration": "adrak",
            "english_translation": "ginger",
            "category": "spices",
            "variations": ["adrak", "adrakh"]
        },
        {
            "hindi_transliteration": "lehsun",
            "english_translation": "garlic",
            "category": "spices",
            "variations": ["lehsun", "lasun"]
        },
        {
            "hindi_transliteration": "haldi",
            "english_translation": "turmeric",
            "category": "spices",
            "variations": ["haldi", "haladi"]
        },
        {
            "hindi_transliteration": "mirch",
            "english_translation": "chili",
            "category": "spices",
            "variations": ["mirch", "mirchi"]
        },
        {
            "hindi_transliteration": "jeera",
            "english_translation": "cumin",
            "category": "spices",
            "variations": ["jeera", "jira"]
        },
        {
            "hindi_transliteration": "dhaniya",
            "english_translation": "coriander seeds",
            "category": "spices",
            "variations": ["dhaniya", "dhania"]
        },
        {
            "hindi_transliteration": "roti",
            "english_translation": "bread",
            "category": "grains",
            "variations": ["roti", "rotli"]
        },
        {
            "hindi_transliteration": "dal",
            "english_translation": "lentils",
            "category": "legumes",
            "variations": ["dal", "daal", "dhal"]
        },
        {
            "hindi_transliteration": "paneer",
            "english_translation": "cottage cheese",
            "category": "dairy",
            "variations": ["paneer", "panner"]
        },
        {
            "hindi_transliteration": "dahi",
            "english_translation": "yogurt",
            "category": "dairy",
            "variations": ["dahi", "daahi"]
        },
        {
            "hindi_transliteration": "ghee",
            "english_translation": "clarified butter",
            "category": "dairy",
            "variations": ["ghee", "ghi"]
        }
    ]
    
    # Create embeddings for all entries and variations with proper mapping
    all_texts = []
    embedding_to_entry_map = []
    
    for entry_idx, entry in enumerate(knowledge_base):
        # Add main transliteration
        all_texts.append(entry["hindi_transliteration"])
        embedding_to_entry_map.append(entry_idx)
        
        # Add variations
        for variation in entry["variations"]:
            all_texts.append(variation)
            embedding_to_entry_map.append(entry_idx)
        
        # Add English translation for reverse lookup
        all_texts.append(entry["english_translation"])
        embedding_to_entry_map.append(entry_idx)
    
    # Generate embeddings
    embeddings = embedding_model.encode(all_texts)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    vector_index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    vector_index.add(embeddings.astype('float32'))
    
    # Save knowledge base, index, and mapping
    kb_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge_base.pkl')
    index_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector_index.faiss')
    map_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'embedding_map.pkl')
    
    with open(kb_path, 'wb') as f:
        pickle.dump(knowledge_base, f)
    faiss.write_index(vector_index, index_path)
    with open(map_path, 'wb') as f:
        pickle.dump(embedding_to_entry_map, f)

def retrieve_relevant_context(query, top_k=5):
    """Retrieve relevant context from the knowledge base."""
    global embedding_model, vector_index, knowledge_base, embedding_to_entry_map
    
    if embedding_model is None or vector_index is None:
        return []
    
    # Generate query embedding
    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Search for similar entries
    scores, indices = vector_index.search(query_embedding.astype('float32'), top_k * 2)  # Get more results to filter
    
    # Map indices back to knowledge base entries using the proper mapping
    relevant_entries = []
    seen_entries = set()
    
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(embedding_to_entry_map):
            entry_idx = embedding_to_entry_map[idx]
            if entry_idx not in seen_entries and entry_idx < len(knowledge_base):
                relevant_entries.append(knowledge_base[entry_idx])
                seen_entries.add(entry_idx)
                
                if len(relevant_entries) >= top_k:
                    break
    
    return relevant_entries

def find_best_match(word, relevant_context, threshold=0.6):
    """Find the best match for a word using fuzzy matching."""
    best_match = None
    best_score = 0
    
    for entry in relevant_context:
        # Check exact matches first
        if word.lower() == entry["hindi_transliteration"].lower():
            return entry, 1.0
        
        for variation in entry["variations"]:
            if word.lower() == variation.lower():
                return entry, 1.0
        
        # Check fuzzy matches
        score = SequenceMatcher(None, word.lower(), entry["hindi_transliteration"].lower()).ratio()
        if score > best_score and score >= threshold:
            best_score = score
            best_match = entry
        
        for variation in entry["variations"]:
            score = SequenceMatcher(None, word.lower(), variation.lower()).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = entry
    
    return best_match, best_score

def construct_prompt(query, relevant_context):
    """Construct a prompt for the LLM with relevant context."""
    context_str = ""
    for entry in relevant_context:
        context_str += f"- {entry['hindi_transliteration']} ({', '.join(entry['variations'])}) → {entry['english_translation']} ({entry['category']})\n"
    
    prompt = f"""You are a multilingual spell corrector and translator. Your task is to:
1. Detect if the input contains Hindi words written in English letters
2. Correct any misspellings
3. Translate Hindi words to English
4. Return the result in JSON format

Context (Hindi-English translations):
{context_str}

Input query: "{query}"

Please analyze the query and return a JSON response with the following format:
{{
    "original": "{query}",
    "corrected": "corrected and translated version",
    "detected_language": "hindi/english/mixed",
    "translations": [
        {{"hindi": "word", "english": "translation"}}
    ]
}}

Response:"""
    
    return prompt

@rag_bp.route('/correct', methods=['POST'])
@cross_origin()
def correct_query():
    """Main endpoint for query correction and translation."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Initialize models if not already done
        if embedding_model is None:
            initialize_models()
        
        # Retrieve relevant context
        relevant_context = retrieve_relevant_context(query)
        
        # Use improved translation logic with caching
        result = improved_translation_logic_with_caching(query, relevant_context)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in correct_query: {str(e)}")
        return jsonify({'error': str(e)}), 500

def is_likely_hindi_word(word):
    """Check if a word is likely to be Hindi based on common patterns."""
    word_lower = word.lower()
    
    # Common English words that should not be treated as Hindi
    common_english_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'this', 'that', 'these', 'those', 'it', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'can', 'may', 'might', 'must', 'shall', 'hello', 'world', 'water', 'food',
        'good', 'bad', 'big', 'small', 'new', 'old', 'hot', 'cold', 'yes', 'no',
        'time', 'day', 'night', 'man', 'woman', 'child', 'home', 'work', 'school',
        'car', 'bus', 'train', 'plane', 'book', 'pen', 'paper', 'computer', 'phone'
    }
    
    if word_lower in common_english_words:
        return False
    
    # Known Hindi words from our translation dictionary
    hindi_words = {
        'doodh', 'paal', 'palak', 'bhindi', 'karela', 'methi', 'dhaniya', 'pudina',
        'adrak', 'lehsun', 'haldi', 'mirch', 'jeera', 'gajar', 'gobi', 'lauki',
        'baingan', 'aloo', 'tamatar', 'pyaaz', 'kela', 'aam', 'paneer', 'dahi',
        'ghee', 'makhan', 'malai', 'lassi', 'chaas', 'paani', 'jal', 'chawal',
        'dal', 'daal', 'roti', 'chapati', 'naan', 'atta', 'besan', 'suji',
        'masala', 'garam', 'namak', 'cheeni', 'gud', 'shahad', 'tel', 'sarson',
        'til', 'kothimbir', 'sabzi', 'curry', 'biryani', 'pulao', 'raita',
        'chutney', 'achar', 'papad', 'samosa', 'pakora', 'kachori', 'paratha',
        'khana', 'bhojan', 'nashta', 'dopahar', 'raat', 'paisa', 'rupaya',
        'ghar', 'makaan', 'bazar', 'dukan', 'kitab', 'kalam', 'kagaz',
        'aankhein', 'haath', 'pair', 'sar', 'baal', 'muh', 'naak', 'kaan',
        'safed', 'kaala', 'laal', 'peela', 'neela', 'hara', 'gulabi', 'bhura',
        'accha', 'bura', 'bada', 'chota', 'lamba', 'chouda', 'patla', 'mota',
        'thand', 'meetha', 'namkeen', 'kadva', 'khatta', 'aur', 'ke', 'saath',
        'mein', 'main', 'ki', 'ka', 'kya', 'hai', 'hain', 'kaise', 'kahan',
        'kab', 'kyun', 'koi', 'kuch', 'sab', 'yeh', 'woh', 'yahan', 'wahan'
    }
    
    if word_lower in hindi_words:
        return True
    
    # Common Hindi word patterns
    hindi_patterns = ['aa', 'ee', 'oo', 'ai', 'au', 'kh', 'gh', 'ch', 'jh', 'th', 'dh', 
                     'ph', 'bh', 'sh', 'rh', 'lh', 'nh', 'mh']
    
    # Common Hindi word endings  
    hindi_endings = ['ak', 'al', 'an', 'ar', 'at', 'ay', 'el', 'en', 'er', 'et', 'ey', 
                    'i', 'il', 'in', 'ir', 'it', 'iy', 'o', 'ol', 'on', 'or', 'ot', 'oy',
                    'u', 'ul', 'un', 'ur', 'ut', 'uy', 'aa', 'ee', 'ii', 'oo', 'uu']
    
    # Check for double letters (common in Hindi transliteration)
    has_double_letters = any(word_lower[i] == word_lower[i+1] for i in range(len(word_lower)-1))
    
    # Check for common Hindi patterns
    has_hindi_pattern = any(pattern in word_lower for pattern in hindi_patterns)
    
    # Check for common Hindi endings
    has_hindi_ending = any(word_lower.endswith(ending) for ending in hindi_endings)
    
    # Check length (Hindi words are often longer than 3 characters)
    is_reasonable_length = len(word_lower) > 3
    
    # Check for specific Hindi characteristics
    has_hindi_char_pattern = any(char in word_lower for char in 'dqxz') == False  # These are rare in Hindi
    
    # Score based on various factors
    score = 0
    if has_double_letters: score += 1
    if has_hindi_pattern: score += 2
    if has_hindi_ending: score += 1
    if is_reasonable_length: score += 1
    if has_hindi_char_pattern: score += 1
    
    return score >= 2

def improved_translation_logic_with_caching(query, relevant_context):
    """Improved translation logic with simple translation and caching."""
    words = query.split()
    translated_words = []
    translations = []
    detected_language = "english"
    
    # Common Hindi words mapping
    common_hindi_words = {
        "aur": "and",
        "ke": "of",
        "saath": "with",
        "mein": "in",
        "main": "in",
        "ki": "of",
        "ka": "of",
        "kya": "what",
        "hai": "is",
        "hain": "are",
        "kaise": "how",
        "kahan": "where",
        "kab": "when",
        "kyun": "why",
        "koi": "any",
        "kuch": "some",
        "sab": "all",
        "yeh": "this",
        "woh": "that",
        "yahan": "here",
        "wahan": "there"
    }
    
    for word in words:
        translated = False
        original_word = word
        
        # Remove punctuation for matching
        clean_word = word.strip('.,!?;:')
        
        # Check common Hindi words first
        if clean_word.lower() in common_hindi_words:
            translated_words.append(common_hindi_words[clean_word.lower()])
            translations.append({
                "hindi": clean_word,
                "english": common_hindi_words[clean_word.lower()]
            })
            detected_language = "hindi" if detected_language == "english" else "mixed"
            translated = True
        else:
            # Try to find best match in knowledge base
            best_match, score = find_best_match(clean_word, relevant_context)
            
            if best_match and score > 0.6:  # Only use matches with good confidence
                translated_words.append(best_match["english_translation"])
                translations.append({
                    "hindi": clean_word,
                    "english": best_match["english_translation"]
                })
                detected_language = "hindi" if detected_language == "english" else "mixed"
                translated = True
            else:
                # Use simple translation for unknown words
                if is_likely_hindi_word(clean_word):
                    try:
                        # Translate using simple method
                        translated_word = translate_word_simple(clean_word, 'hi', 'en')
                        
                        if translated_word != clean_word.lower():
                            translated_words.append(translated_word)
                            translations.append({
                                "hindi": clean_word,
                                "english": translated_word
                            })
                            detected_language = "hindi" if detected_language == "english" else "mixed"
                            translated = True
                            
                            # Add to knowledge base for future use
                            add_to_knowledge_base(clean_word, translated_word, "auto_translated")
                    except Exception as e:
                        print(f"Translation error for '{clean_word}': {e}")
        
        if not translated:
            # Keep original word if no translation found
            translated_words.append(original_word)
    
    corrected = " ".join(translated_words)
    
    return {
        "original": query,
        "corrected": corrected,
        "detected_language": detected_language,
        "translations": translations,
        "relevant_context": [
            {
                "hindi": entry["hindi_transliteration"],
                "english": entry["english_translation"],
                "category": entry["category"]
            } for entry in relevant_context
        ]
    }

@rag_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'RAG Multilingual Translator'})

@rag_bp.route('/cache/stats', methods=['GET'])
@cross_origin()
def cache_stats():
    """Get translation cache statistics."""
    return jsonify({
        'cache_size': len(translation_cache),
        'knowledge_base_size': len(knowledge_base) if knowledge_base else 0
    })

@rag_bp.route('/cache/clear', methods=['POST'])
@cross_origin()
def clear_cache():
    """Clear the translation cache."""
    global translation_cache
    translation_cache = {}
    save_translation_cache()
    return jsonify({'message': 'Cache cleared successfully'})

# Initialize models when the module is imported
try:
    initialize_models()
except Exception as e:
    print(f"Warning: Failed to initialize models at startup: {e}")
    print("Models will be initialized on first request.")

