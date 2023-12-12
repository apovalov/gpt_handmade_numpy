# GPT SEO

# Transformers
As we said, GPT (Generative Pre-trained Transformer) is a family of models like Transformers. Transformers have reshaped the current landscape of machine learning models for Natural Language Processing (NLP - Natural Language Processing), which led to the emergence of models such as GPT or BERT (we had a separate task about it, related to analyzing the tone of feedback). Today, we will focus on one of the most attractive applications of Transformers: text generation.

The machine learning revolution associated with Transformers started with the article "Attention is all you need" in 2017. It wasn't just a new algorithm or method, it was a radically new way of thinking about ordered data (which includes text as a string of words or characters). This shift had a huge impact on all areas of machine learning (even beyond natural language), making Transformers the foundation of modern machine learning.

What makes Transformers so good?
Previously, recurrent networks (RNNs), models like LSTM and GRU, were used to process sequences (words or characters). These had limited speed and relatively short memory. Transformers, on the other hand, process a gigantic number of words at once and understand the context (meaning of the text) more deeply.

Other benefits of Transformers include:

1. Self-Attention. Previously, in RNN, the attention mechanism (Attention) had a supporting role. Transformers, on the other hand, uses Attention as a key way of exchanging "meaning" between words in a text to understand the "overall context". We will soon become familiar with the Attention mechanism when we realize it with our own hands.
2. Parallelism. In recurrent models, word processing proceeds sequentially (one after the other), while in Transformers, processing of words in text proceeds simultaneously (due to the addition of positional coding, which we will discuss later). Due to this parallelism, processing is fast, and the meaning of words that were somewhere at the beginning of the sequence is not lost. Due to these two factors, Transformers scale beautifully, gaining new features and increasing the depth of contextual understanding.
3. Transfer Learning (Few-Shot Learning). Instead of starting to train a new specialized language model for a specific task, large language models like Transformers can be used in Few-Shot Learning mode: by showing by examples what data is input in our specific task, what format of data we expect in the output, the rest of the work will be done by the model. This greatly extends the applicability of Transformers to the widest pool of tasks. We will use this property today to develop an SEO assistant using GPT-2.

A token is a character, part of a word, or a whole word into which the input text is divided.

The input text is often called a prompt.


![Alt text](/img/image.png)

Typically, in English text, a token is about 3/4 of a word (750 words contain about 1000 tokens). For other languages this proportion may be different (for example, in Russian there are much more tokens per "kilometer of text", which makes processing and generation slower). Why this happens is a topic for a separate task and we will not go into it here.

## Byte Pair Encoding (BPE)

BPE is a tokenization method that enables efficient text encoding. It's a cornerstone algorithm in GPT-2's tokenization process, among others.

### Purpose of BPE

The goal of BPE is to merge frequently occurring characters or sequences of characters (byte pairs) into single tokens, while rare sequences are split into individual characters. This method allows for representing a text dataset with a minimal number of tokens, enhancing the expressiveness of the vocabulary and making the most out of the available dimensional space.

### Concept Behind BPE

The concept behind Byte Pair Encoding is as follows:

1. Initially, the "vocabulary" consists of individual characters. This base allows encoding any text by subdividing it into available characters, which is the most granular level of information representation.

2. When a pair of tokens occurs frequently, we add this pair to the dictionary. Upon encountering this pair again, it's considered a single new token. This step incrementally builds the vocabulary from frequent pairs of characters.

3. The operation is repeated. At this stage, we combine any frequent pair of tokens into a single token, regardless if they are single or multiple characters, thus creating tokens composed of multiple characters due to their frequent co-occurrence.

4. Iterations continue until the dictionary reaches a predefined size limit.

This approach to tokenization is particularly powerful for large-scale language modeling as it strikes a balance between the granularity of characters and the semantic richness of full words.

## GPT Token Generation Explained

The Generative Pretrained Transformer (GPT) model generates tokens sequentially. It takes a sequence of tokens as input and predicts one token at a time extending the sequence. Here's how it works:

1. **Token Generation:** At each timestep, GPT outputs a new token, expanding the input sequence. The process continues until a special <EOS> (End of Sequence) token is generated, which indicates the end of the output sequence.

2. **Understanding Tokens:** To understand how GPT generates tokens, it's essential to grasp the training objective of GPT. The model is trained to predict the next token in a sequence, learning complex patterns in language from a vast corpus of text data.

3. **Softmax Layer:** The predictions are made using a softmax layer, which converts the logits (raw output from the last layer of the model) into probabilities.

    ```python
    softmax(z_i) = exp(z_i) / sum(exp(z_j) for j in range(len(z)))
    ```

    The logits are the model's raw predictions before normalization by the softmax function, which turns them into probabilities summing to 1.

4. **Choosing Tokens:** The simplest method for token selection is to choose the most probable next token, known as "greedy sampling". This approach, however, often results in less creative and more predictable text.

By updating the model parameters through backpropagation, the model refines its ability to predict more accurately, making the generated sequences coherent and contextually relevant.


![Alt text](/img/image-1.png)

Sampling: Top-k & Top-p
In the output of the model, we actually get a whole probability table (token → its probability). Technically, we use token IDs as keys (together with the model we have a token dictionary that converts tokens to IDs and back). For simplicity, we will assume that we have a list of tokens List[str] and the index of a token corresponds to its ID.

Instead of taking top-1 from the mentioned probability table each time, we can take a subsample of the most probable tokens and generate one of them with a probability directly proportional to the probability of encountering the given word in the text (according to the model's prediction). This is how we arrive at sampling, namely, the top-k mode (usually k is taken in the neighborhood of 50).

![Alt text](/img/image-2.png)

The other, more dynamic sampling option is called top-p. Here p is the probability. For example top_p=0.75 says that we will limit sampling to the minimum set of tokens that provides at least 75% of all probabilities. At the expense of this, at the moment when the model is less sure which word to take next, its pool of candidates will be wider. If it is very sure among which tokens to choose, the number of candidate tokens will be smaller.

Translated with www.DeepL.com/Translator (free version)

![Alt text](/img/image-3.png)

Top-k and top-p sampling can be combined (usually top-k is applied first, narrowing the candidate pool, normalizing the probabilities so that they sum to one again, then top-p is applied).

## Sampling: Temperature
Finally, let's get acquainted with another crucial parameter, Temperature, with which we can control the confidence of the model and the creativity of its generation (which is crucial for creative product descriptions in SEO optimization). The higher the temperature, the more "creative" its responses (more chaotic, more random). The lower it is, the more the language model is dominated by "cold reasoning" and rational precise thinking.

How does temperature change the probability distribution of occurrence of tokens? - It "sharpen" them (increasing the probability of probable tokens and decreasing the probability of unlikely tokens) or "flatten" them (moving them to a more uniform distribution).


![Alt text](/img/image-4.png)

## Temperature in Softmax Function

Numerically, temperature is a hyperparameter that affects the distribution of probabilities across logits:

- **Standard Mode (Temperature = 1.0):** This is the default setting. Probabilities remain the same as initially predicted by the model.

- **Creative Mode (Temperature > 1.0):** Setting the temperature higher, e.g., 1.5 or 2.0, increases the chances of selecting less likely tokens, encouraging diversity and creativity in the generated text.

- **Cautious Mode (Temperature < 1.0):** A lower temperature, e.g., 0.5 or 0.75, steers the selection towards more likely tokens, leading to more predictable and conservative outputs.

The temperature is applied to the softmax function as shown below:

```python
softmax(x_i / T)
```



How does GPT see the words?
### 1. Embeddings of tokens.
As we know, machine learning models, including Transformers, can't work with words in their raw form. They need to turn words into "numerical meanings", which we call embeddings. Technically, an embedding is a vector of very high dimensionality. For example, in GPT-2, the dimensionality of embeddings is between 768 and 1600 numbers:

![Alt text](/img/image-5.png)

GPT-2 exists in four different sizes: the number of layers and the dimensionality of the embeddings (as well as the total number of parameters) are all that technically distinguish the models of this family from each other. The larger the model, the better the quality of text generation.

During training, Transformer "learns" not only the parameters of each layer, but also the embeddings of the tokens - before they enter the model itself. This part of the model is called the Encoder. In content, it is a table - literally like a dictionary in Python, only it has vectors as values. We can use it to get each token_id to get its embedding.

So, for example, GPT-2 has a dictionary size of 50,257 tokens. For the smallest model (which has embeddings of dimension 768), the dimension of the embedding table will be 50,257 x 768. As we already know from the previous step, the dictionary and its size, i.e., what tokens the text is split into, is set in advance using an algorithm separate from the model, the tokenizer.

![Alt text](/img/image-6.png)


What does such a vector representation of words give us? The space of embeddings should be perceived as an N-dimensional map of meanings: in this space, words close in meaning (more specifically, tokens) have similar embeddings (in the sense of some measure of proximity, e.g., scalar product).


### 2. Position coding
Unlike recurrent neural networks (LSTM, GRU), Transformers process all tokens simultaneously. But how is this achieved? After all, there is coherence, there is consistency in text: closely spaced words are usually connected by context and often relate to each other.

The way to achieve this is ingeniously simple: let's first "inject" position information into the embedding of each token (which will help the model recover the original word order in the text) and then run them through Transformer. After that, we can even shuffle all the input tokens (keeping their embeddings already containing position information) - and their processing will not change (if we turn a blind eye to Masked Self-Attention procedures for now).

![Alt text](/img/image-7.png)


### 3. The intuition behind Transformers.
Finally, what does the Transformer itself do with these "meanings"? The heart of Transformers is the Self-Attention mechanism, which we'll deal with in detail in the next steps. What it does: it is essentially the shearing of information between tokens ("meaning exchange"). The second element of the architecture is FFN (Feed Forward Network), which on the one hand occupies 80% of the model's weights, but on the other hand is perceived as less important than the attention mechanism itself.

So, each Transformer block alternates between 2 stages:

Self-Attention is the "exchange of meanings", the refinement of embeddings with a look back at the context.
FFN is the processing of each token's embedding separately, in isolation from the others.
Metaphor
If you've worked with big data (Hadoop, PySpark), you've heard of the MapReduce concept. The FFN and Self-Attention transformer layers are similar to Map and Reduce transformations. Mapper is similar to FFN and transforms some of the data on its own. Reducer takes the Mapper results and combines, accumulates, aggregates. A complex Data Pipeline can be visualized as a succession of such Map and Reduce phases. Similarly, a complex text transformation (a game with meanings) can be reproduced with Transformers.


Let's go through the logic again: Self-Attention adds to each token's embedding the embeddings of other tokens with different weights (usually neighboring tokens, due to the position encoding we discussed above). Trace FFN transforms this hodgepodge into something more consistent and meaningful. Through Attention, each token "listens" to advice from all the others, giving different attention to different sources. In FFN, the token "decides" how to update its viewpoint based on this advice piled together, as well as its own opinion (its own embedding).

As we move from layer to layer, each embedding from the token's isolated "sense" + position information becomes more and more "context-absorbed", transforming into context-dependent embedding.

![Alt text](/image-9.png)