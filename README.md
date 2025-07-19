# Regional Shopping AI

A sophisticated AI-powered shopping assistant that combines Retrieval-Augmented Generation (RAG) with real-time product search capabilities. The application provides intelligent shopping recommendations, maintains shopping lists, and offers multilingual support for regional shopping experiences.

## 🚀 Features

### Core Functionality
- **RAG-based Q&A System**: Intelligent question-answering using vector embeddings and semantic search
- **Smart Shopping Lists**: AI-powered shopping list management with product recommendations
- **Real-time Product Search**: Integration with multiple search engines for live product data
- **Multilingual Support**: Automatic language detection and translation capabilities
- **Vector Similarity Search**: FAISS-powered semantic search for accurate product matching

### API Endpoints
- **RAG System** (`/api/rag/`): Question-answering with context-aware responses
- **Shopping Management** (`/api/shopping/`): Shopping list creation and product search
- **User Management** (`/api/users/`): User registration and profile management

### Frontend Interfaces
- **Main Dashboard** (`index.html`): Primary user interface
- **Shopping Demo** (`shopping_demo.html`): Interactive shopping list demonstration
- **Real Shopping** (`real_shopping.html`): Live product search interface

## 🛠️ Technology Stack

### Backend
- **Flask**: Web framework with CORS support
- **SQLAlchemy**: Database ORM for user management
- **Sentence Transformers**: Text embedding generation
- **FAISS**: Vector similarity search and indexing
- **Transformers**: Hugging Face models for NLP tasks
- **PyTorch**: Deep learning framework

### AI/ML Components
- **Vector Embeddings**: Semantic text representation
- **Similarity Search**: Content-based recommendation system
- **Language Detection**: Automatic language identification
- **Translation Services**: Multilingual support

### Data Storage
- **SQLite**: User data and application state
- **Pickle Files**: Serialized ML models and embeddings
- **FAISS Indices**: Optimized vector search structures

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/shanthan1999/regional-shopping-ai.git
   cd regional-shopping-ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   Create a `.env` file in the root directory:
   ```env
   SECRET_KEY=your_secret_key_here
   # Add other environment variables as needed
   ```

5. **Initialize the application**
   ```bash
   python src/main.py
   ```

The application will be available at `http://localhost:5000`

## 🔧 Configuration

### Environment Variables
- `SECRET_KEY`: Flask application secret key
- Additional API keys for external services (if required)

### Data Files
The application uses several data files stored in `src/data/`:
- `knowledge_base.pkl`: RAG knowledge base
- `vector_index.faiss`: FAISS vector index
- `shopping_items.pkl`: Product database
- `embedding_map.pkl`: Embedding mappings

## 🚀 Usage

### RAG System
Send POST requests to `/api/rag/ask` with questions to get AI-powered responses based on the knowledge base.

### Shopping Lists
- Create and manage shopping lists via `/api/shopping/`
- Get product recommendations based on semantic similarity
- Search for real-time product information

### User Management
- Register users via `/api/users`
- Manage user profiles and preferences

## 🏗️ Project Structure

```
regional-shopping-ai/
├── src/
│   ├── data/                 # Data files and indices
│   ├── database/            # SQLite database
│   ├── models/              # Data models
│   ├── routes/              # API endpoints
│   ├── services/            # Business logic services
│   ├── static/              # Frontend files
│   └── main.py              # Application entry point
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## 🔍 API Documentation

### RAG Endpoints
- `POST /api/rag/ask`: Submit questions for AI responses
- `GET /api/rag/status`: Check system status

### Shopping Endpoints
- `POST /api/shopping/list`: Create shopping lists
- `GET /api/shopping/search`: Search products
- `POST /api/shopping/recommend`: Get recommendations

### User Endpoints
- `GET /api/users`: List all users
- `POST /api/users`: Create new user
- `GET /api/users/<id>`: Get user details
- `PUT /api/users/<id>`: Update user information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- Facebook AI Research for FAISS
- Flask community for the web framework
- Contributors and testers

## 📞 Support

For support, please open an issue on GitHub or contact the development team.

---

**Regional Shopping AI** - Transforming the shopping experience with artificial intelligence.