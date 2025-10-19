# Ultralytics Code Assistant.

This project is a simple **RAG (Retrieval-Augmented Generation)** code assistant built to answer questions about the **Ultralytics YOLO Python source code**, specifically the `models`, `engine`, and `data` directories.

This version runs the **Qwen/Qwen2.5-Coder-7B-Instruct** model locally on your own NVIDIA GPU.

---

## 🚀 Core Features

- **Chat Interface** — Simple Q&A using Gradio  
- **Code Indexing** — Processes `.py` files from the `ultralytics/` directory  
- **Vector Search** — Uses MongoDB Atlas for storing and searching code chunks  
- **Answer Generation** — Run the Qwen-Coder model locally  

---

## ⚙️ Hardware Requirements

Running a 7B parameter model on specific hardware.

- **GPU:** NVIDIA GPU 
- **VRAM:** At least 24 GB recommended  
- **CUDA:** Proper CUDA drivers compatible with torch versions  

---

## 🧩 Setup and Running

### 1. Project Setup

```bash
# 1. Create repository path
mkdir yolo-code-repo

# 2. Clone your project repository
cd yolo-code-repo

# 3. Clone the Ultralytics source code
# This is required for the indexing script to find the files
git clone https://github.com/ultralytics/ultralytics.git

# 4. Return base path
cd ..

# 5. Create a 'uv' virtual environment
uv venv
source .venv/bin/activate

# 6. Install dependencies
# This step may take a while as it downloads required libraries (with CUDA)
uv pip sync
```

---

### 2. MongoDB Atlas Setup

1. Create a free (M0) account at **[MongoDB Atlas](https://www.mongodb.com/atlas)**.  
2. Create a new **Database** (e.g., `yolo_code_db`) and a **Collection** (e.g., `code_chunks`).  
3. Create a **Vector Search Index**:

   - Navigate to your `code_chunks` collection → **Search Indexes** tab.  
   - Click **Create Search Index → JSON Editor**.  
   - Use **default** as the Index Name and paste the following configuration (for 768-dimension Jina embeddings):

```json
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "embedding": {
        "dimensions": 768,
        "similarity": "cosine",
        "type": "knnVector"
      }
    }
  }
}
```

---

### 3. Environment Variables

Copy the example environment file and update it:

```bash
cp .env.example .env
# Edit .env and add your MONGO_URI
```

Your `.env` file should contain:

```bash
MONGO_URI="mongodb+srv://<your_username>:<your_password>@<your_cluster_address>/"
```

---

### 4. Step 1: Index the Code (Silent Mode)

This script processes the YOLO codebase and uploads it to MongoDB Atlas.  
It runs silently but shows `tqdm` progress bars.

```bash
python index_code.py
```

---

### 5. Step 2: Launch the Application

Once indexing is complete, start the Gradio app:

```bash
python app.py
```

When you run this, your console will display logs and loads the model onto your GPU.  
Once loading completes, the interface will be available at:

👉 [http://127.0.0.1:7860](http://127.0.0.1:7860)
