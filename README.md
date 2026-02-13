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

## Data Analysis

To test embedding sensitivity to semantic changes, I modified three classmate descriptions:
1. **Miguel Palafox**: Changed "enjoy" and other positive verbs to "detest" (Major semantic inversion).
2. **Nikita Neveditsin**: Added a distinct clause about "avoiding my family at all costs" (Context addition).
3. **Jeevan Dhakal**: Rephrased existing hobbies, e.g., changing "watching movies" to "viewing films" (Phrasing change).

**Impact Analysis:**
The results showed varying degrees of impact on the embeddings:
*   **Miguel Palafox** experienced a large drop in cosine similarity (**0.548**). This drastic change confirms that the model effectively captured the complete inversion of sentiment from positive to negative.
*   **Nikita Neveditsin** showed a moderate drop (**0.836**), reflecting the introduction of completely new information while maintaining the original context.
*   **Jeevan Dhakal** retained a relatively high similarity (**0.862**). Even though the specific words changed, the semantic proximity of terms like "films" vs "movies" kept the vector close to the original.

These results suggest that the embeddings are robust to widely varying phrasing but are appropriately sensitive to strong sentiment inversions and the injection of new context.
