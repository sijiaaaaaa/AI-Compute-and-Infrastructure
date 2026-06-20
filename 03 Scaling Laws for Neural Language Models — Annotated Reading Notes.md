# Scaling Laws for Neural Language Models — Annotated Reading Notes

Paper: *Scaling Laws for Neural Language Models*
Authors: Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei
arXiv: 2001.08361

> This repository contains my personal annotated reading notes on the paper, with a focus on translating neural scaling laws into compute infrastructure and financial trade-off frameworks.

---


## Section 1 — Cross-Entropy Loss

| Paper Reference                                                                                                                                     | My Notes                                                                                                                                                                                                                                                                                                        |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Topic:** The paper studies language model performance using cross-entropy loss.                                                                   | Cross-entropy loss measures how much probability the model assigns to the correct next token. Lower loss means the model is better at predicting real text. In finance terms, this becomes the technical output metric we can compare against compute input.                                                    |
| **Formula:** <br><br> $\text{Loss} = -\log P(\text{correct next token} \mid \text{previous tokens})$                                                | The negative sign converts “higher probability is better” into “lower loss is better.” This allows training to be framed as minimizing error. Without the negative sign, the objective would be to maximize log probability, which is mathematically valid but not usually how ML loss functions are expressed. |
| **Training vs. generation:** During training, the next token is already known. During generation, the model has to choose the next token by itself. | This is the key distinction. Cross-entropy loss is used during training and evaluation, when the correct next token comes from the dataset. Top-k, top-p, greedy decoding, and temperature are generation-time strategies, not training loss functions.                                                         |

### 2.1 Next-token prediction as training examples

Suppose the training text is:

```text
I drink a cup of coffee every morning.
```

During training, the model turns this sentence into many next-token prediction tasks:

| Input previous tokens           | Correct next token |
| ------------------------------- | ------------------ |
| `I`                             | `drink`            |
| `I drink`                       | `a`                |
| `I drink a`                     | `cup`              |
| `I drink a cup`                 | `of`               |
| `I drink a cup of`              | `coffee`           |
| `I drink a cup of coffee`       | `every`            |
| `I drink a cup of coffee every` | `morning`          |

So during training, the correct answer is known because it comes from the training data.

---

### 2.2 Why next-token prediction is a classification problem

Next-token prediction is not binary classification. It is a large multi-class classification problem.

If the vocabulary has 50,000 tokens, then at each position the model is choosing among 50,000 possible next tokens.

Example:

```text
I drink a cup of ___
```

The correct next token is:

```text
coffee
```

The true label can be represented as a one-hot vector, while the model outputs a probability distribution:

| Token    | True label $p_i$ | Model probability $q_i$ |
| -------- | ---------------: | ----------------------: |
| `coffee` |                1 |                    0.70 |
| `tea`    |                0 |                    0.20 |
| `water`  |                0 |                    0.05 |
| `car`    |                0 |                  0.0001 |
| `table`  |                0 |                  0.0001 |
| `...`    |                0 |                     ... |

The “0 and 1” here do not mean there are only two possible outcomes. They mean that, for this specific training example, the correct token has label 1 and all other tokens have label 0.

---

### 2.3 Cross-entropy formula

The full cross-entropy formula is:

$$
H(p, q) = -\sum_i p_i \log(q_i)
$$

Where:

* $p_i$ = true distribution
* $q_i$ = model-predicted distribution
* $i$ = each token in the vocabulary

Using the same example, the true token is `coffee`.

Substitute the values:

$$
H(p, q) = -[1 \cdot \log(0.70) + 0 \cdot \log(0.20) + 0 \cdot \log(0.05) + 0 \cdot \log(0.0001) + 0 \cdot \log(0.0001)]
$$

Because all incorrect tokens have true label 0, those terms disappear:

$$
H(p, q) = -\log(0.70)
$$

So in the one-hot label case, cross-entropy simplifies to:

$$
\text{Loss} = -\log P(\text{correct token})
$$

This is why cross-entropy loss and negative log-likelihood are often used almost interchangeably in language modeling.

---

### 2.4 Why the negative sign is used

The negative sign is not just for making the number positive. The more important reason is:

> We want to convert “higher probability is better” into “lower loss is better.”

Machine learning training usually minimizes a loss:

$$
\min \text{Loss}
$$

If we did not use the negative sign, the objective would be:

$$
\max \log(P)
$$

That is mathematically valid, but ML training is usually framed as minimizing loss.

So we multiply by $-1$:

$$
\text{Loss} = -\log(P)
$$

Example:

$$
-\log(0.8) = 0.223
$$

$$
-\log(0.2) = 1.609
$$

The model that gives the correct token 0.8 probability has lower loss than the model that gives it 0.2 probability.

---

### 2.5 Average loss across a short sentence

Example sentence:

```text
I drink coffee
```

Assume we calculate two prediction steps:

| Step | Previous tokens | Correct next token | Model A probability |
| ---: | --------------- | ------------------ | ------------------: |
|    1 | `I`             | `drink`            |                0.60 |
|    2 | `I drink`       | `coffee`           |                0.20 |

Step 1:

$$
P(\text{drink} \mid \text{I}) = 0.60
$$

$$
-\log(0.60) = 0.511
$$

Step 2:

$$
P(\text{coffee} \mid \text{I, drink}) = 0.20
$$

$$
-\log(0.20) = 1.609
$$

Average loss formula:

$$
\text{Average Loss} = \frac{1}{T}\sum_{t=1}^{T} -\log P(x_t \mid x_{<t})
$$

In this example, $T = 2$:

$$
\text{Average Loss} = \frac{1}{2}[-\log P(\text{drink} \mid \text{I}) + -\log P(\text{coffee} \mid \text{I, drink})]
$$

Substitute the probabilities:

$$
\text{Average Loss} = \frac{1}{2}[-\log(0.60) + -\log(0.20)]
$$

$$
\text{Average Loss} = \frac{1}{2}[0.511 + 1.609]
$$

$$
\text{Average Loss} = 1.060
$$

So Model A’s average cross-entropy loss is:

$$
1.060
$$

---

### 2.6 Compare with a stronger model

Now suppose Model B assigns higher probability to the correct tokens:

$$
P(\text{drink} \mid \text{I}) = 0.90
$$

$$
P(\text{coffee} \mid \text{I, drink}) = 0.70
$$

Then:

$$
\text{Average Loss} = \frac{1}{2}[-\log(0.90) + -\log(0.70)]
$$

$$
\text{Average Loss} = \frac{1}{2}[0.105 + 0.357]
$$

$$
\text{Average Loss} = 0.231
$$

Comparison:

| Model   | $P(\text{drink} \mid \text{I})$ | $P(\text{coffee} \mid \text{I, drink})$ | Average Loss |
| ------- | ------------------------------: | --------------------------------------: | -----------: |
| Model A |                            0.60 |                                    0.20 |        1.060 |
| Model B |                            0.90 |                                    0.70 |        0.231 |

Model B has lower loss, which means it assigns higher probability to the true tokens in the training text.

---

<img width="767" height="344" alt="image" src="https://github.com/user-attachments/assets/e01b1c8e-8fda-4228-ba80-334dfeaa266c" />


