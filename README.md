# Maximal-Cone-NMF

_Python implementation of paper:_  
[1] Veeranna Rupashree, K. V., & Pimentel-Alarcón, D. L.  "A Maximal-cone solution to NMF" (2026).  
---

## 📂 File Structure  

📁 NoSE-NMF/  
│  
├── main_real_data.ipynb # Main notebook to run the NMF pipelines for real data  
├── main_synthetic.ipynb # Main notebook to run the NMF pipelines for synthetic data  
├── maximal_cone_nmf.py # Runs NoSEs algorithm to find all N NoSEs  
├── matrix_utils.py # Helper functions for generating synthetic dataset  
├── requirements.txt # Python dependencies  
└── README.md # Project documentation  

---

## ⚙️ How to Run

### 📦 Prerequisites

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## 🧪 Run Example
### 🔬 1. Synthetic Data
- Open main_synthetic.ipynb
- Run the the cells with custom m, n and r.

### 📊 2. Real Data
To use your own dataset:  
🛠️ Step 1: 
- Open main_nose_nmf.ipynb
- Modify the second cell to load your data.

▶️ Step 2: Run in Notebook
- Run the reamining cells to execute on your custom dataset.


