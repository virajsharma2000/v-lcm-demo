---

# 🧠 Large Concept Model - Sentence Continuation using SONAR + BLASER

This Python script uses Facebook AI's [SONAR](https://github.com/facebookresearch/SONAR) and [BLASER](https://github.com/facebookresearch/SONAR/tree/main/sonar/models/blaser) models to predict sentence continuation by computing semantic similarity between a prompt sentence and a dataset of full sentences.

---

## 📦 Requirements

* Python 3.8+
* PyTorch
* `sonar` package (from [facebookresearch/SONAR](https://github.com/facebookresearch/SONAR))

---

## 📁 Files

* `training_dataset.json`: A list of sentences, one per array, e.g.

  ```json
  [
    ["The cat sat on the mat."],
    ["They visited Europe during summer."]
  ]
  ```
* `large_concept_model.py`: Main script to compute sentence similarity and extract the continuation.

---

## 🚀 How It Works

1. **Load Models**
   Loads BLASER 2.0 and the SONAR basic text embedding encoder.

2. **Input Prompt**
   A fixed sentence prompt (e.g., `"cat sat on"`).

3. **Find Most Similar Sentence**

   * Computes similarity scores (1 to 5) between the prompt and each sentence in the dataset.
   * Picks the sentence with the **highest similarity**.

4. **Extract Next Words**
   From the most similar sentence, it extracts the **next few words** not already present in the prompt.

---

## 🧪 Example Output

```bash
{4.02: 0, 2.15: 1, 1.97: 2}
['the', 'mat.']
```

---

## 📝 Customize Prompt

You can change the prompt in the script:

```python
prompt_sentence = ['cat sat on']
```

---

## 📌 Notes

* Language is set to `"eng_Latn"` for English.
* This script performs **sentence-level reasoning**, not word-level generation.
* Works best with varied and rich training data.

---

## 🤖 Credits

* [SONAR](https://github.com/facebookresearch/SONAR) by Facebook AI
* BLASER 2.0: For semantic similarity scoring

---
