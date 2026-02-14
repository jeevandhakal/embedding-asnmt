## Embedding Sensitivity Tests

I tested how much the similarity rankings change when using different embedding models.  
I compared **all-MiniLM-L6-v2** and **all-mpnet-base-v2**.

For each model, I created embeddings from classmates’ descriptions. Then I calculated cosine similarity between my embedding and every classmate. Based on these scores, I ranked classmates from most similar to least similar.

To measure how similar the two ranking lists are, I used **Spearman’s rank correlation**.  
The result was:

**Spearman rho = 0.8578**

This value is close to 1, which means the rankings from the two models are very similar.

For my results (**Zilong Wang**):

**Top-10 Model 1 (all-MiniLM-L6-v2):**

- Binziya Siddik
- Nikola Kriznar
- Jayanta Sarker Shuva
- Md Musfiqur Rahman
- Somto Muotoe
- Md Riad Arifin
- Miguel Palafox
- Jeevan Dhakal
- Sridhar Vadla
- Mohammad Pakdoust

**Top-10 Model 2 (all-mpnet-base-v2):**

- Nikola Kriznar
- Md Musfiqur Rahman
- Binziya Siddik
- Somto Muotoe
- Jayanta Sarker Shuva
- Mohammad Pakdoust
- Md Riad Arifin
- Krushi Mistry
- Soundarya Venkataraman
- Pawan Lingras

**Top-10 overlap: 7 classmates**

Overlapping classmates:

- Binziya Siddik
- Nikola Kriznar
- Jayanta Sarker Shuva
- Md Musfiqur Rahman
- Somto Muotoe
- Md Riad Arifin
- Mohammad Pakdoust

Overall, the high Spearman rho and strong overlap show that the rankings are mostly stable.  
Changing the embedding model causes only small differences.  
This means the system is not highly sensitive to model choice.

**How to run the program?**

```bash
python model_comparison.py --emb1 embeddings_minilm.json --emb2 embeddings_mpnet.json --me "Zilong Wang"
```
