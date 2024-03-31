# Randeng-T5-text-intelligent-pre-annotation
Using Randeng-T5 for Text Intelligent Pre annotation Algorithm
**Introduction of algorithms**

Google T5-Chinese improved model: Randeng-T5. (Chinese ZERO-SHOT list Rank 1 model), pre-labeling can choose 77M, 784M two kinds of parametric quantities of large models.

Improved based on the Google T5 model, the model is based on Transformer's encoder-decoder architecture, and the Chinese-English corresponding vocabulary and embeddings are re-trained on the Wudao Corpora large corpus, and the model is pre-trained by using Corpus-Adaptive Pre- By using Corpus-Adaptive Pre-Training (CAPT), it is more suitable for intelligent pre-labeling of Chinese data.

**Innovation Points:**
1. Supports multi-task learning, which can efficiently process sentiment analysis, news classification, text categorization, intent recognition, natural language reasoning, semantic matching, multiple choice, denotative disambiguation, extractive reading comprehension, entity recognition, keyword extraction, keyword recognition, generative summarization, a total of 13 tasks at the same time, which reduces the need for multiple independent models.
2、Supporting the control of pre-labeled text, some predefined keywords and phrases are used to guide the model to generate more accurate prediction results.
3、Support for multi-granularity modeling of Chinese natural language, which can simultaneously consider different levels of information such as character level, word level, sentence level, etc., so as to improve the performance of the model, which is more suitable for Chinese data.
