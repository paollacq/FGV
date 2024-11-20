# Blockchain Anomaly Detection

This project implements a pipeline for detecting anomalies in Bitcoin transactions using unsupervised machine learning techniques. The detection is based on features such as total amounts sent and received by addresses and net transactional flow, using datasets containing historical Bitcoin transactions.

---

## **Dataset**
The project uses the Bitcoin transaction dataset from 2011 to 2013 available at the IEEE DataPort:

- [Bitcoin Transactions Data 2011–2013](https://ieee-dataport.org/open-access/bitcoin-transactions-data-2011-2013)

The dataset contains information on Bitcoin transactions, including:
- **from_address:** The sender's address hash.
- **to_address:** The receiver's address hash.
- **timestamp:** The date and time of the transaction.
- **value:** The value of the transaction in Bitcoin.

---

## **Base Article**
The methodology and approach used in this project are inspired by the following research article:
- [Fraud Detection in Blockchains Using Machine Learning](https://ieeexplore.ieee.org/document/9094045)

---

## **Project Pipeline**
1. **Preprocessing:**
   - Reads the raw dataset in chunks (to handle large files efficiently).
   - Cleans the data by removing duplicates and missing values.
   - Extracts features such as:
     - `total_sent`: Total value sent by each address.
     - `total_received`: Total value received by each address.
     - `net_flow`: Difference between received and sent values.

2. **Anomaly Detection:**
   - Uses **Isolation Forest** to identify outliers in the data based on the generated features.
   - Scores transactions based on how anomalous they appear compared to normal patterns.

3. **Visualization:**
   - Plots the distribution of anomaly scores for normal and anomalous transactions to assess model performance.

---

## **Steps to Run**
1. **Install Dependencies:**
   Make sure you have Python installed and run:
   ```bash
   pip install -r requirements.txt
