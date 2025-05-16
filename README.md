# ATS AI Screener – Backend

This is the backend API server for the ATS AI Screener project, built with FastAPI and SQLAlchemy. It provides RESTful endpoints to support the frontend application, handling resume processing, AI-powered matching, user authentication, and feedback management.

---

## Development Setup

### Prerequisites:

* Python 3.9 or newer
* pip (Python package manager)
* Git (for cloning the repository)

---

## Installation

### Clone the repository:

```bash
git clone https://github.com/Divyey/RAG-AI-Screening-Backend.git
cd RAG-AI-Screening-Backend/app
```

### Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### Install dependencies:

```bash
pip install -r ../requirements.txt
```

---

## Environment Variables

Create a `.env` file in the `app/` directory to configure environment-specific variables such as:

```env
DATABASE_URL=sqlite:///./app.db
SECRET_KEY=your_secret_key
OPENAI_API_KEY=your_openai_api_key
```

Make sure to update `.env.example` (if provided) with the required variables.

---

## Running the Server

Start the development server with hot reload:

```bash
uvicorn main:app --reload --port 8080
```

The API will be available at `http://localhost:8080`.

---

## Project Structure

```
app/
├─ alembic/                  # Database migrations
├─ app.db                    # SQLite database file (if using SQLite)
├─ main.py                   # FastAPI application entrypoint
├─ embeddings.py             # Embedding utilities
├─ vectorstore_faiss.faiss   # Vector store index file
├─ uploads/                  # Uploaded resume files
├─ models/                   # Database models (if separated)
├─ routers/                  # API route modules (if separated)
├─ dependencies.py           # Dependency injections
├─ requirements.txt          # Python dependencies
└─ ...
```

---

## Useful Scripts

* `uvicorn main:app --reload --port 8080`  - Start development server with auto-reload
* `alembic upgrade head`        - Run database migrations
* `pytest`                     - Run tests (if tests are implemented)

---

## Features and Middleware

* RESTful API endpoints for resume upload, scoring, matching, and feedback
* User authentication and role-based access control
* Integration with OpenAI API for AI-powered matching
* Vector search using FAISS
* Database migrations with Alembic
* Middleware including CORS, error handling, and logging

---

## Documentation

API documentation is automatically generated and available at:

```
http://localhost:8080/docs
```

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## License

This project is licensed under the MIT License.

---

## Related Repositories

* **Frontend**: [https://github.com/Divyey/RAG-AI-Screening-Frontend](https://github.com/Divyey/RAG-AI-Screening-Frontend)

---

*Last updated: May 16, 2025*
