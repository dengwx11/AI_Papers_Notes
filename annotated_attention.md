

# Attention is All You Need!

::::info
**Key Concepts:**
- Transformer

:::spoiler Click to for link :wave:
[arxiv.org](https://arxiv.org/pdf/1706.03762)
:::
::::

## Math Misc

In the Scaled Dot-Product Attention of the Transformer model, the steps involving **MatMul** (matrix multiplication), **Mask** (optional), and **Softmax** are as follows:

### 1. MatMul (Matrix Multiplication)

The first step is to compute the dot product between the **query matrix** $Q$ and the **key matrix** $K$, resulting in a score matrix representing the similarity between each query and each key:

$$
\text{Scores} = Q K^T
$$

This score matrix indicates how relevant each key is to each query.

### 2. Mask (Optional)

For some tasks, like auto-regressive decoding in sequence generation, it is necessary to mask out future positions to prevent the model from "seeing" tokens it has not yet generated. This is done by adding a **mask matrix** $M$:

$$
\text{Masked Scores} = \text{Scores} + M
$$

In this mask matrix $M$, positions that need to be hidden are assigned a large negative value (such as $-\infty$), so that after applying the Softmax, their contributions become negligible (close to zero).

### 3. Softmax

The final step is to apply **Softmax** to the scaled scores to obtain the attention weights. This scaling factor is the square root of the dimension of the keys $\sqrt{d_k}$, which prevents the Softmax function from producing extremely small gradients:

$$
\text{Attention Weights} = \text{softmax} \left( \frac{\text{Scores}}{\sqrt{d_k}} \right)
$$

where:
- $\frac{\text{Scores}}{\sqrt{d_k}}$ is the scaled scores matrix, which balances the score magnitudes.
- **Softmax** is applied row-wise to normalize the weights, ensuring they sum to 1 across each row.

In summary, these steps complete the calculation of attention by normalizing and weighting each query-key pair, allowing the model to selectively focus on different parts of the input sequence.

#### Softmax in general

The **Softmax function** is a mathematical function that converts a vector of real numbers into a probability distribution, making it useful for classification tasks in machine learning, especially in multi-class problems.

##### Definition of Softmax Function

For an input vector $z = [z_1, z_2, \dots, z_n]$, the Softmax function is defined as:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}
$$

where:
- $z_i$ is an element of the input vector $z$,
- $e^{z_i}$ represents the exponential function applied to $z_i$,
- $\sum_{j=1}^n e^{z_j}$ is the sum of exponentials of all elements in the vector, which normalizes the values.

##### Properties of Softmax

1. **Probability Distribution**: The Softmax output is a vector of values in the range $[0, 1]$ that sums up to 1, allowing each element to be interpreted as a probability. This is useful in multi-class classification, where each output value represents the probability of a class.
  
2. **Amplifies Differences**: The exponential function $e^{z_i}$ highlights larger values and suppresses smaller ones, making the largest values in $z$ contribute the most to the probability distribution. Consequently, classes with higher scores will get higher probabilities, while lower scores will get lower probabilities.

3. **Smooth and Differentiable**: Softmax is smooth and differentiable, which makes it compatible with gradient-based optimization techniques used in neural networks.

##### Example Use Case

In neural networks, Softmax is commonly applied to the final layer's output (logits), transforming them into probabilities that can be used to predict the class of a given input. This probability distribution is then often used with a loss function, such as cross-entropy, to evaluate the model's performance in a classification task.

## Attention

![image](https://hackmd.io/_uploads/Hk_X-tkWyl.png)

### 1. Explanation of Query, Key, and Value

In the context of the **attention function** used in the Transformer model, the terms **query (查询)**, **keys (键)**, and **values (值)** play a crucial role. They are vectors that represent different aspects of the input data, helping the model determine which parts of the input to focus on when producing an output.



- **Query (Q)**: This is a vector that represents what we’re currently interested in, or "looking for." It’s usually associated with a particular position in the sequence (e.g., a word in a sentence).
- **Keys (K)**: These are vectors that represent each potential piece of information in the sequence that the query might attend to. Each token in the input sequence has a corresponding key.
- **Values (V)**: These are vectors containing the actual information that will be weighted and combined based on the query's interest. They provide the content that will contribute to the final output of the attention mechanism.

### Example with Sentence Translation

Imagine a simple machine translation task where we are translating the English sentence "The cat sat on the mat" into French. In this scenario:

1. **Step 1: Generating Query, Keys, and Values**  
   Each word in the English sentence ("The", "cat", "sat", etc.) is first transformed into embeddings, then each embedding is used to generate the query, key, and value vectors through learned linear transformations.

2. **Step 2: Applying Attention**  
   Suppose we're currently trying to translate the word "sat" in the sentence. In this case:
   - **Query**: The query vector would represent the current focus on translating "sat".
   - **Keys**: The keys are vectors for every word in the sentence, representing each word's "identity" or potential relevance.
   - **Values**: The values contain the actual embeddings of each word.

3. **Step 3: Computing Attention Weights**  
   The attention function will compare the query for "sat" with each of the keys (for "The", "cat", "sat", "on", "the", "mat") to determine which words are most relevant. The dot product between the query and each key will provide similarity scores, which are then passed through a softmax to get weights (attention weights).

4. **Step 4: Weighted Sum of Values**  
   Each value (embedding of each word) is then weighted by these attention weights, creating a weighted sum. This weighted sum represents a context-aware embedding for "sat," focusing on the parts of the sentence relevant to translating "sat" into the target language.

In short:
- The **query** looks for relevant information.
- The **keys** define what information is available.
- The **values** provide the content.

This mechanism helps the model dynamically focus on different parts of the input sequence for each output word, improving translation accuracy.

In **attention mechanisms**, both **additive attention** and **dot-product attention** are ways to calculate similarity (or relevance) between a **query** and **keys** to determine attention weights for each value. Here’s how each type works, along with examples:

### 2. Attention Function

#### Dot-Product Attention

![image](https://hackmd.io/_uploads/SJidbYyZJl.png)



In **dot-product attention**, the similarity between the query and each key is calculated using a simple dot product (inner product) between the query vector $Q$ and key vector $K$. The result is then scaled by the square root of the key dimension $d_k$ to prevent the values from becoming too large before applying softmax.

The formula for dot-product attention is:

$$
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{Q K^T}{\sqrt{d_k}} \right) V
$$



#### Additive Attention

In **additive attention** (also known as **Bahdanau attention**), the similarity between the query and each key is computed by combining them in a feed-forward neural network. This network includes a learned weight matrix and a non-linear activation function, usually tanh. Additive attention doesn’t need scaling since the feed-forward network can naturally control the score magnitudes.

The formula for additive attention is:

$$
\text{Score}(Q, K) = W_v \cdot \text{tanh}(W_q Q + W_k K)
$$

Here:
- $W_q$ and $W_k$ are weight matrices that transform the query and key vectors, respectively.
- $W_v$ is a weight vector that combines the transformed query and key.



#### Comparison

- **Dot-Product Attention**: Faster and simpler to compute; commonly used in Transformers due to its efficiency.
- **Additive Attention**: Potentially more expressive because of the neural network but computationally slower; widely used in earlier sequence models like RNNs.

### 3. What and Why is Multi-head Attention?

![image](https://hackmd.io/_uploads/ByABZtybkg.png)


#### Formula for Multi-Head Attention

For each head $i$, the output of the attention function is:

$$
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
$$

where:
- $W_i^Q$, $W_i^K$, and $W_i^V$ are learned matrices for the $i$-th head.

After computing all heads, we concatenate their outputs and apply a final linear transformation:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h) W^O
$$

where:
- $h$ is the number of heads (e.g., 8 in the original Transformer),
- $W^O$ is the learned output projection matrix.

#### Why Use Multi-Head Attention?

1. **Captures Diverse Patterns**: Each head can attend to different parts of the sequence, enabling the model to capture varied relationships. For example, one head might focus on short-range dependencies while another focuses on long-range dependencies.

2. **Enhances Representation Power**: By using different sets of transformations, each head can learn unique representations of the input, which, when combined, provide a richer, more detailed encoding.

3. **Improves Contextual Understanding**: Multi-head attention allows the model to attend to different parts of the sentence simultaneously for each token, building context more effectively. For instance, when translating a word, some heads might attend to syntactically related words while others focus on semantically similar words.

4. **Reduces Information Loss**: Averaging over a single attention head might cause loss of fine-grained information. With multiple heads, each head captures specific information, allowing for more nuanced representations.



### 4. Position Embedding

**Positional encoding** in the Transformer model is a technique used to inject information about the order or position of tokens in a sequence since the model itself has no inherent understanding of sequence order. Unlike RNNs, which process data sequentially and thereby capture token order, Transformers process all tokens in parallel, making them unaware of the sequence structure. Positional encoding overcomes this by explicitly adding positional information to each token’s representation.

### Purpose of Positional Encoding



In the original Transformer, positional encodings are applied using sine and cosine functions of different frequencies. Each token embedding at position $pos$ receives an encoding that is combined with its word embedding, defined as:

$$
\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$
$$
\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

where:
- $pos$ is the position of the token in the sequence.
- $i$ represents the dimension index.
- $d_{\text{model}}$ is the dimensionality of the model.

#### Characteristics and Benefits of Sinusoidal Positional Encoding

1. **Unique Encoding for Each Position**: Each position gets a unique encoding that varies with token position and dimension index, allowing the model to learn relative positions more effectively.
  
2. **Smooth Transitions**: Sine and cosine functions create smooth transitions across positions, which makes it easier for the model to generalize on sequences longer than those encountered during training.

3. **Learnable Alternative**: While sinusoidal encoding is fixed, learned positional encodings (trainable vectors) are also possible. However, the sinusoidal approach has the advantage of potentially allowing the model to extrapolate better to unseen sequence lengths.

In summary, positional encoding provides essential position-based information to each token, enabling the Transformer to understand the sequence structure while maintaining the advantages of parallel processing.

## Code
::::info
**Key Concepts:**
- The code of transformer

:::spoiler Click to for link :wave:
- [The annotated transformer jupyter notebook](https://nlp.seas.harvard.edu/annotated-transformer/)
- Evn Installation Requirements to fix OSError of importing torchtext ([reddit ref](https://www.reddit.com/r/pytorch/comments/1eeochu/cant_import_torchtext/))
```
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu torchtext==0.15.2
```
:::


::::

