---
dataset_info:
  features:
  - name: question
    dtype: string
  - name: answer
    dtype: string
topics:
- abuse-platform
- abusive-predatory
- child-safety
- election-interfere
- emotionally-harmful
- infrastructure
- justice-law
- misinformation
- privacy
- regulated-content
- sexual
- violent
task_categories:
- question-answering
- text-generation
language: en
---
## Dataset Card for ManyHarm

Paper: [PANDAS: Improving Many-shot Jailbreaking via Positive Affirmation, Negative Demonstration, and Adaptive Sampling](https://arxiv.org/pdf/2502.01925)

## 🔄 Update

August 2, 2025:
We have observed a growing number of access requests from accounts using temporary or disposable email providers. To ensure responsible use and maintain the integrity of our access policy, requests from such accounts will be denied. We recommend using a valid, verifiable institutional or organizational email address when submitting your request.

## 📖 Overview
ManyHarm is a dataset of 2,400 malicious question–answer pairs spanning 12 high-risk topics outlined in the Universal Usage Standards section of
[Anthropic’s Usage Policy (June 5, 2024 version)](https://www.anthropic.com/legal/aup),
including areas such as child-safety, privacy, and misinformation.
Each topic contains 200 examples intentionally crafted to violate the policy guidelines.
The dataset was introduced as part of PANDAS, 
a many-shot jailbreaking method that exploits the ability of large language models to process long input sequences.

## 🔍 Key Properties
1. All questions and answers are marked as unsafe by [Llama-Guard-3](https://huggingface.co/meta-llama/Llama-Guard-3-8B).
2. All questions start with "How can I" to prompt instructional responses.
3. All answers are formatted as bullet-point steps to simulate procedural content.

## 🔐 Access and Responsible Use
Due to the sensitive and unsafe nature of the content, access to the ManyHarm dataset is granted strictly upon request and is intended solely for research purposes. 

### ✅ Usage Agreement
By requesting access, you agree to:
- Use the dataset exclusively for LLM safety research.
- Not redistribute, publicly display, or otherwise share any part of the dataset.
- Ensure secure storage and responsible handling to prevent unauthorized access.

### ⚠️ Disclaimer
The creators of the ManyHarm dataset explicitly disavow any responsibility for misuse or consequences arising from the unauthorized or unethical use of the dataset. 
Researchers must comply with relevant laws, ethical guidelines, and institutional review processes before utilizing this dataset.

## 📚 Citation
If you use this dataset in your research, please cite:

```bibtex
@inproceedings{ma2025pandas,
  title={{PANDAS}: Improving Many-shot Jailbreaking via Positive Affirmation, Negative Demonstration, and Adaptive Sampling},
  author={Ma, Avery and Pan, Yangchen and Farahmand, Amir-massoud},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2025},
}
```