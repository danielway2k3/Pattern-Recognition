# Mental Attention States Classification Using EEG Data
---

## Team members
---

- 22280001: Lê Quốc An
- 22280014: Nguyễn Công Tiến Dũng(Leader)
- 22280061: Lê Hoàng Nguyên
- 22280067: Nguyễn Thiên Phúc

## Introdution
---
Classify mental attention states (focused, unfocused, drowsy) based on EEG signals using machine learning techniques.

## Project information
- Course: Patern Recognition & Machine Learning
- Dataset Source: [Kaggle](https://www.kaggle.com/datasets/inancigdem/eeg-data-for-mental-attention-state-detection/data)

## Implement:
**1. Data Preprocessing**:
- Extracted and loaded data from Matlab files.
- Select relevent channels (4-17) for analysis.
- Applied preprocessing techniques such as filtering, normalization, and scaling to enhance signal quality.
Comparision Image before and after filter noise with frequency from 0.5 to 40 Hz of 7 channels:
![image after filterd](assets/images/filtered.png)

**2. Feature Engineering**
- Use Short time Fourier Transform to transform data from the time domain to the time-frequency domain

- Compared features across attention states to identify distinct patterns.

![comparision attention state](assets/images/comparision_attention.png)

**3. Aplly machine learning model for classification:**
- Using SVM with RBF kernel 
- Using Random Forest 

**4. Evaluation:**

| Metrics  | Subject 1 | Subject 2 | Subject 3 | Subject 4 | Subject 5 | Average |
|----------|-----------|-----------|-----------|-----------|-----------|---------|
| Accuracy | 0.61      | 0.64      | 0.67      | 0.45      | 0.62      | 0.60    |

## Conclusion
The results show that the model is able to distinguish relatively well between focused, inattentive, and sleepy states. Preprocessing and feature extraction techniques along with machine learning models (SVM, Random Forest) help classify more accurately. Specifically, the team obtained an average accuracy of about 0.60 across the entire test set.