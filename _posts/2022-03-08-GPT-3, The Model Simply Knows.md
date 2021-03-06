---
toc: true
layout: post
description: GPT-3 is one of the greatest leaps in the area of NLP. In this post, we will try to go through the basics of the algorithm and how it is working. 
categories: [GPT-3, NLP, Research Paper]
title: GPT-3, The Model Simply Knows!
image: images/GPT-3-preview.gif
---


![]({{ site.baseurl }}/images/GPT-3.gif "GPT-3, The Model Simply Knows")

Natural Language Processing (NLP) has been one of the most challenging areas in deep learning. This is due to several reasons. First, human language is complicated, even for humans themselves! 

Consider asking someone about his experience in learning Chinese, for example. Without a doubt, he will tell you that this is difficult. The difficulty in learning any language is that almost all the meaning can be derived from the contextual conversational pipeline. In other words, you can’t tell what the purpose. Let’s see the following example:

- The UN sanctioned Iran’s use of nuclear power (allowed)
- The UN agreed on sanctions for Iran because they used nuclear power (punishment)
This example is callous even for a native English speaker, and it would be tougher for a model to make some kind of reasoning for these two sentences. 

Second, language is handled as sequences that can vary in input or output length, which is another obstacle for the modeling process. Because the model simply does not know what is the most important word in a sequence that could predict the next works in a conversational pipeline. 

Now, let’s see some of the revolutionary approaches in NLP that made a real leap!

## Attention is All You Need
This [paper](https://arxiv.org/abs/1706.03762) was a turning point in all NLP. It solves most of the context problems. That means we can finally solve the problem of the most essential or related past words in predicting the next ones using a structured mathematical description. 

![]({{ site.baseurl }}/images/attention.png "Transformers Structure")


This paper laid the foundation for the use of *Transformers* (with attention heads) instead of *Sequence Models* (RNNs) only.

Afterward, these concepts paved the way towards a new breed of language models that are capable of reasoning in some sense. 

## BERT 

BERT which stands for Bidirectional Encoder Representations from
Transformers found in this [paper]( https://arxiv.org/abs/1810.04805 ) put the idea of transformers with attention heads at the top of language models. BERT was, breathtakingly, a model that marks a new era in language models. The idea behind BERT is simple as it constructs out of two main stages.

- Pre-training, in which we just train our model on a large corpus to do some supervised tasks like next sentence prediction ( NSP ). 
- Fine Tuning, another training step using a relatively smaller dataset to a more specialized task.  

![]({{ site.baseurl }}/images/bert.png "BERT")

BERT, with its 345 million parameters, was able to achieve very superior performance over any model of its kind. 

## GPT-3 

A gigantic model, the largest model ever built by humans. GPT-3, presented by OpenAI in this [paper](https://arxiv.org/abs/2005.14165), is the state-of-the-art (SOTA) language model with a very superior model that is capable of doing the most NLP-related tasks.

Now, we will talk about different aspects of the model from the training data, the number of parameters, and the training procedures. 


### Training Datasets

*GPT-3* is trained on almost the entire internet or, according to my expression, trained on the whole human civilization. 

The corpus is the largest ever collected for any language model ever. It’s kind of mind-blowing to be thinking about how enormous it is! The following table from the original paper depicts the datasets and their weight in the training mix. 

![]({{ site.baseurl }}/images/training.png "Training Data")

These datasets would sum up to 500 billion tokens, which is incredibly massive and incomparable to any other language model. 

In a nutshell, these datasets are forming the whole human knowledge! Almost every webpage you have seen in your life is included in this training process!
Let’s move on to the training procedure.

### Training

We are waiting for OpenAI to reveal more details about the training infrastructure and model implementation. But to put things into perspective, the GPT-3 175B model required 3.14E23 FLOPS of computing for training. Even at theoretical 28 TFLOPS for V100 and lowest three years reserved cloud pricing we could find, this will take 355 GPU-years and cost $4.6M for a single training run. Similarly, a single RTX 8000, assuming 15 TFLOPS, would take 665 years to run.

Time is not the only enemy. The 175 Billion parameters need 700GB memory to store. This is one order of magnitude larger than the maximum memory in a single GPU (48 GB of Quadro RTX 8000). To train the larger models without running out of memory, the OpenAI team uses a mixture of model parallelism within each matrix multiply and model parallelism across the layers of the network. All models were trained on V100 GPU’s on the part of a high-bandwidth cluster provided by Microsoft.

The following graph is showing some information about the training power required to train different sizes of language models in Petaflop/s-days.

![]({{ site.baseurl }}/images/training-time.png "Training Time")


### Model Architecture 

GPT-3 comes in eight sizes, ranging from 125M to 175B parameters. The largest GPT-3 model is an order of magnitude larger than the previous record-holder, T5-11B. The smallest GPT-3 model is roughly the size of BERT-Base and RoBERTa-Base.

All GPT-3 models use the same attention-based architecture as their GPT-2 predecessor. The smallest GPT-3 model (125M) has 12 attention layers, each with 12x 64-dimension heads. The largest GPT-3 model (175B) uses 96 attention layers, each with 96x 128-dimension heads.

GPT-3 expanded the capacity of its GPT-2 by three orders of magnitudes without significant modification of the model architecture — just more layers, wider layers, and more data to train it on.

### Learning Philosophy

Unlike all previous models that are trained on next word prediction and then be fine-tuned on a specific task, GPT-3 is only trained on next word prediction only with no other fine-tuning on any task. However, GPT-3 is doing surprisingly well! Let’s dive into more details about this!

![]({{ site.baseurl }}/images/learning.png "Learning")

As the comparison shows, the model is merely doing almost any task just by remembering it. Some tasks can be achieved and in an excellent performance with no remembering at all. These are zero-shot learning tasks. Others will require one example to follow, and those can be called one-shot learning tasks. Besides, the most difficult tasks that may require several cases to remember are called few-shot learning tasks. 

Despite that this approach is straightforward, it’s useful! I think that every natural language task in the human brain is handled that way. For example, programmers, after spending some time in programming they can immediately think of code as a text completion in their minds, and that’s all. 


The effect of this simple approach did not prove itself before. Only when having the required training data and the enormous model, it become crystal clear!




## Applications

This may be the most breathtaking part about GPT-3. In the following passages, I will try to view most of these applications.

### Text Generation

This is GPT's rockstar application -- a conditional generative model that creates near-human level quality text content. Given the beginning of some articles, the model is asked to generate the rest of the story in a word-by-word fashion.

More precisely, GPT-3 is presented with a title, a subtitle, and the prompt word "Article:" It then writes short articles (~200 words) that fools humans most of the time. According to OpenAI's user study, "mean human accuracy at detecting articles that were produced by the 175B parameter model was barely above change at ~52%". Meaning humans will make random guesses while asking to detect GPT-3 generated articles. In contrast, the mean human accuracy at detecting particles produced by the smallest GPT-3 model (125M) is 76%.

This can be a big deal — "simply" increasing the size of the model by three orders of magnitude can change something that is half-working into something non-distinguishable from human work. In plain English, this empirically shows that the number of model parameters, the FLOP/s-days, and the number of training examples needs to grow according to a power function of the improvement of the model.

Of course, GPT-3 may still produce non-factual content (such as suggesting the popular U.S. TV program "The Tonight Show" is hosted by Megyn Kelly instead of Jimmy Fallon), nor did OpenAI claim the model is ready for writing the last two books of "A Song of Ice and Fire." Nonetheless, getting closer to the finishing line of the Turing test for writing short articles is significant, and will no doubts have an enormous impact on our social media.

### General NLP Tasks

Although writing a new article is excellent, the killer feature of GPT-3 is the ability to be 're-programmed' for general NLP tasks without any finetuning. This is where OpenAI's real ambition lies: having a model to do just about anything by conditioning it with a few examples.

The paper showed a dozen of downstream tasks, ranging from the usual players such as machine translation and question and answer to the unexpected new tasks such as arithmetic computation and one-shot learning of new words. Instead of reiterating the details of each task, the rest of this article will discuss some common patterns across the board.


In the next lines, we will try to cover some of these cool applications.

### Explaining Idioms 

Just by using a few-shot learning approach, the model is surprisingly perfect in explaining idioms! See the conversation below, generated by GPT-3. 

![]({{ site.baseurl }}/images/explain.png "Explaining Idioms")

### Mind-blowing search engine.

It’s not like any search engine because you can ask it in plain English back in plain English also with no search results, only the right and exact answer!

![]({{ site.baseurl }}/images/search-engine.gif "Search Engine")


### Layout Generator

Imagine writing some English sentences and get back in return, a full functioning front-end app in a blink of an eye! 


![]({{ site.baseurl }}/images/layout.gif "Layout Generator")



### The debate of GPT-3 against another GPT-3!

What if you can see an entirely generated debate just be making a GPT-3 model talk with another! It’s incredible. What is more frightening is that these two models after five messages began to the limitations of human beings. 

![]({{ site.baseurl }}/images/debate.png "Debate")


Finally, and without a doubt, GPT-3 has changed the face of language modeling from now on. This superior performance will open the door to dozens of philosophical and ethical questions. Some people may wonder about some jobs, such as will these models make some jobs disappear like writers, coders, and teachers. Is it ethical to use this model to pass exams?! ( Note that GPT-3 was able to score 57% in the SAT exam ). I think the future will answer all of these questions!


One more thing that made me can’t wait for the next GPT is that it will support interactive podcasts! And this is really beyond imagination!

