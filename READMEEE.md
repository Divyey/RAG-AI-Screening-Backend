# ATS AI Screener – Backend

This is the FastAPI backend for the ATS AI Screener project.

## Development

python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
uvicorn main:app --reload


## Environment Variables

Create a `.env` file for secrets, database URLs, etc.

## Database

- Uses SQLite by default.
- Alembic for migrations.

Setup Guides
setup-frontend.md

# Setup: Frontend

## 1. Clone the repository

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name/frontend


## 2. Install dependencies

npm install


## 3. Start the development server


npm run dev


The app will be available at `http://localhost:5173` (or similar, check terminal output).

## 4. Build for production

npm run build

setup-backend.md

# Setup: Backend

## 1. Clone the repository

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name/app


## 2. Create and activate a virtual environment

python -m venv .venv
source .venv/bin/activate


## 3. Install dependencies
pip install -r ../requirements.txt


## 4. Run the FastAPI server

uvicorn main:app --reload


The API will be available at `http://localhost:8080`.

## 5. Database

- By default, uses SQLite.
- To run migrations (if using Alembic):

alembic upgrade head