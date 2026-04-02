# Contributing to DATN

Welcome! This project involves heavy data processing and machine learning models. To keep the repository manageable, we only track the source code in Git. Follow the steps below to set up your local environment.

## 1. Prerequisites
- Python 3.9+
- Node.js & npm (for the frontend)

## 2. Environment Setup

### Backend (Python)
1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Link to requirements file or manually install core packages: faiss-cpu, torch, transformers, fastapi, etc.)*

### Frontend (React/Vite)
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```

## 3. Data Setup (CRITICAL)
The project requires approximately **36GB** of data files that are NOT included in this repository.

1. **Download Data**: Ask the project owner for the share link to the `src/data` and `sample_covers` folders.
2. **Placement**: 
   - Place the data files in `src/data/`.
   - Place image samples in `sample_covers/`.
3. **Verification**: Run `python api.py` to ensure the indices load correctly.

## 4. Workflow
- **Branching**: Create a new branch for your feature: `git checkout -b feature/your-feature-name`.
- **Commits**: Keep commits focused and descriptive.
- **Large Files**: **DO NOT** add large data files to Git. If you create new datasets, share them via the external cloud storage instead.
