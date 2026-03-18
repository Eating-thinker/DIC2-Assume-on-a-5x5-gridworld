# 🧭 5x5 GridWorld Value Iteration (Streamlit Demo)

This project implements **Value Iteration** in a 5x5 GridWorld environment and provides an **interactive visualization** using Streamlit.

---

## 🚀 Demo

👉 **Live Demo:**
https://dic2-assume-on-a-5x5-gridworld-gdhwryvxbjszzxyykh5upx.streamlit.app/

---

## 📌 Problem Setting

* Grid size: **5 × 5**
* Start state: **(0, 0)**
* Goal state: **(4, 4)**
* Obstacles:

  * (1,1)
  * (2,2)
  * (3,3)

The agent must find the optimal path from start to goal using **Value Iteration**.

---

## 🧠 Algorithm: Value Iteration

The value function is updated using:

V(s) = maxₐ [ R(s,a) + γ V(s') ]

* γ (discount factor): controls future reward importance
* Deterministic transitions
* Step reward: negative (encourage shorter path)
* Goal reward: positive

The algorithm iteratively updates values until convergence.

---

## 🎯 Features

* ✅ Interactive GridWorld UI
* ✅ Click to set:

  * Start
  * Goal
  * Obstacles
* ✅ Visualized **Value Function V(s)**
* ✅ Visualized **Optimal Policy (arrows)**
* ✅ Shows **optimal path**
* ✅ Adjustable parameters:

  * Discount factor γ
  * Step reward
  * Goal reward
  * Convergence threshold

---

## 🖥️ How to Run Locally

### 1. Install dependencies

```bash
pip install streamlit
```

### 2. Run the app

```bash
streamlit run test.py
```

---

## 📁 Project Structure

```
.
├── test.py          # Main Streamlit application
├── README.md        # Project documentation
└── requirements.txt # Dependencies (optional)
```

---

## 📊 Example Output

* Each grid cell shows:

  * Arrow → best action
  * Value V(s)
* Path is highlighted from start to goal

---

## 💡 Key Learning Points

* Understanding **Markov Decision Process (MDP)**
* Implementing **Value Iteration**
* Policy extraction from value function
* Visualization of reinforcement learning concepts

---

## 👨‍💻 Author

* Course: DIC2
* Assignment: GridWorld Value Iteration

---

## 🔗 Notes

This project demonstrates how reinforcement learning algorithms can be visualized interactively, making abstract concepts more intuitive.
