---
title:  "[ML]Transformer: Attention is All You Need"
date:   2024-10-19 22:50:00 +0800
categories: ML

header:
  image: /assets/images/2024-09-27-Transformer/2024-10-25-23-26-37.png
---

In **sequence-to-sequence (Seq2seq)** problems, the model takes an input sequence and generates an output sequence, with the length of the output being determined by the model itself. This approach is central to tasks like speech recognition, machine translation, and speech-to-speech translation, and is even applicable to conversational AI or chatbot systems.
<table><tr>  <td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-19-23-06-39.png" width="300"> </div></td>  </tr></table>

Most applications in natural language processing (NLP) can be framed as **question-answering** tasks. For example, translation can be seen as answering the question, “What is the translation of this sentence?” Similarly, text summarization can be framed as, “What is the summary of this article?” and sentiment analysis as “Is this sentence positive or negative?” While Seq2seq models offer a generalized approach to question-answering tasks, models optimized with domain-specific knowledge often outperform standard Seq2seq models in practical applications.
<table><tr>  <td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-19-23-17-01.png" width="300"> </div></td>  </tr></table>

Beyond these examples, it is notable that Seq2seq models can also be used for syntactic parsing, where the task is to generate a syntactic tree for a sentence. By representing this tree as a sequence, it too becomes suitable for Seq2seq processing. Additionally, tasks such as multi-label classification and object detection can be approached with Seq2seq models.
<table><tr>  <td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-19-23-31-50.png" width="300"> </div></td>  
 <td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-19-23-32-04.png" width="300"> </div></td> 
</tr></table>

<table><tr> 
 <td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-20-00-11-53.png" width="300"> </div></td>
<td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-20-00-12-08.png" width="300"> </div></td> 
</tr></table>

The Seq2seq model is traditionally composed of an **encoder** and a **decoder**. This architecture was first introduced in the paper _Sequence to Sequence Learning with Neural Networks_ in September 2014, with the **Transformer** model now being the most well-known Seq2seq variant.
<table><tr>  <td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-20-00-16-19.png" width="600"> </div></td>  </tr></table>

## Encoder
The encoder processes an input sequence of vectors and outputs a transformed sequence of vectors. Various architectures exist for this purpose, but the Transformer architecture specifically leverages **Self-Attention**. Unlike simple Self-Attention, the Transformer incorporates a **residual connection**, which combines the output vector with the original input to create a more robust representation. Following this, **layer normalization** is applied to standardize the residual vector’s values. This normalized vector is then passed through a **fully connected network**, again combined with a **residual connection**. A final **layer normalization** completes the encoder’s output.
In the complete Transformer encoder architecture, **positional encoding** is first applied to the input vector to encode the sequential order of tokens. Then, **multi-head self-attention layers** and **Add & Norm operations** (residual connection with layer normalization) are applied, followed by a **feed-forward network** and additional **Add & Norm** operations.

<table><tr>  <td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-25-20-40-16.png" width="300"> </div></td>  
<td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-25-20-46-25.png" width="300"> </div></td> 
<td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-25-20-46-01.png" width="300"> </div></td>
</tr></table>

## Decoder

There are two primary types of decoders, **auto-regressive** and **non-auto-regressive**.
In auto-regressive decoding, as applied in speech recognition, the encoder processes an audio sequence, outputting a vector sequence that is subsequently fed into the decoder. Along with this, the decoder receives tokens, such as a **BEGIN token**. When fed with the BEGIN token, the decoder produces a one-hot vector, which is then transformed into a probability distribution through the soft-max function. The word with the highest score in this distribution is selected as the output. This output word, in vector format, is then fed back into the decoder to predict the next word, producing a sentence token-by-token in sequence.

<table><tr>  <td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-25-21-20-57.png" width="300"> </div></td> 
 <td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-25-21-21-17.png" width="300"> </div></td>  
<td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-25-21-21-32.png" width="300"> </div></td>  
 </tr></table>

The encoder and decoder share structural similarities: both begin with positional encoding, followed by multi-head self-attention, Add & Norm layers, feed-forward networks, and further Add & Norm layers. The key difference in the decoder lies in the use of **masked multi-head self-attention**. This self-attention mechanism allows each output vector to consider only the preceding input vectors and the current one, in result making output tokens generated sequentially.

<table><tr>  
<td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-25-21-25-30.png" width="300"> </div></td>  
 <td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-25-21-28-41.png" width="300"> </div></td> 
</tr></table>

Another important task is that the decoder need to determine the length of the generated sentence and recognize when to stop. This is achieved by training the decoder to recognize a special **END token**, which signals the completion of the sequence.
<table><tr>  <td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-25-21-39-04.png" width="300"> </div></td>  
<td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-25-21-39-13.png" width="300"> </div></td>  
</tr></table>

What an auto-regressive decoder does is that it generates output vectors based on all previously generated output vectors, processing them one-by-one. In contrast, a **non-auto-regressive (NAT) decoder** generates all output vectors simultaneously from the entire set of input vectors, which facilitates parallel processing and simplifies output length control, although it may underperform compared to the auto-regressive approach.
<table><tr>  
<td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-25-21-47-38.png" width="300"> </div></td> 
<td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-25-21-47-51.png" width="300"> </div></td>  
</tr></table>

Between the encoder and decoder lies an essential component known as **cross-attention**, where the decoder’s query vectors interact with the key and value vectors from the encoder. This cross-attention mechanism enables the model to align and generate a final output vector by linking the encoder and decoder representations.
<table><tr>  <td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-25-21-57-27.png" width="300"> </div></td>  
<td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-25-22-04-01.png" width="300"> </div></td>
</tr></table>

# Optimization Seq2Seq/Transformer
- Copy Mechanism
- Guided Attention
- Beam Search
- Scheduled Sampling
<table><tr>  <td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-25-22-33-01.png" width="300"> </div></td>  
<td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-25-22-33-23.png" width="300"> </div></td> 
</tr></table>
<table><tr> 
<td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-25-22-33-49.png" width="300"> </div></td>  
<td><div align=center> <img src="/assets/images/2024-09-27-Transformer/2024-10-25-22-39-46.png" width="300"> </div></td>  
</tr></table>


