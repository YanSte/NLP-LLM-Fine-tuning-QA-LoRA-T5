# | NLP | LLM | Fine-tuning | Chatbot LoRA T5 Large |

## Natural Language Processing (NLP) and Large Language Models (LLM) with Fine-Tuning LLM and make Chatbot with LoRA and Flan-T5 Large

![Learning](https://t3.ftcdn.net/jpg/06/14/01/52/360_F_614015247_EWZHvC6AAOsaIOepakhyJvMqUu5tpLfY.jpg)

# <b><span style='color:#78D118'>|</span> Overview</b>

In this notebook we're going to Fine-Tuning LLM:

<img src="https://github.com/YanSte/NLP-LLM-Fine-tuning-Trainer/blob/main/img_2.png?raw=true" alt="Learning" width="50%">

Many LLMs are general purpose models trained on a broad range of data and use cases. This enables them to perform well in a variety of applications, as shown in previous modules. It is not uncommon though to find situations where applying a general purpose model performs unacceptably for specific dataset or use case. This often does not mean that the general purpose model is unusable. Perhaps, with some new data and additional training the model could be improved, or fine-tuned, such that it produces acceptable results for the specific use case.

<img src="https://github.com/YanSte/NLP-LLM-Fine-tuning-Trainer/blob/main/img_1.png?raw=true" alt="Learning" width="50%">

Fine-tuning uses a pre-trained model as a base and continues to train it with a new, task targeted dataset. Conceptually, fine-tuning leverages that which has already been learned by a model and aims to focus its learnings further for a specific task.

It is important to recognize that fine-tuning is model training. The training process remains a resource intensive, and time consuming effort. Albeit fine-tuning training time is greatly shortened as a result of having started from a pre-trained model. 

<img src="https://github.com/YanSte/NLP-LLM-Fine-tuning-Trainer/blob/main/img_3.png?raw=true" alt="Learning" width="50%">


### The Power of Fine-Tuning: An Overview
Fine-tuning, a crucial aspect of adapting pre-trained models to specific tasks, has witnessed a revolutionary approach known as Low Rank Adaptation (LoRA). Unlike conventional fine-tuning methods, LoRA strategically freezes pre-trained model weights and introduces trainable rank decomposition matrices into the Transformer architecture's layers. This innovative technique significantly reduces the number of trainable parameters, leading to expedited fine-tuning processes and mitigated overfitting.

### What is LoRA?

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/0*kzZ2_LZqBO9_hTi3.png" alt="Learning" width="50%">

LoRA represents a paradigm shift in fine-tuning strategies, offering efficiency and effectiveness. By reducing the number of trainable parameters and GPU memory requirements, LoRA proves to be a powerful tool for tailoring pre-trained large models to specific tasks. This article explores how LoRA can be employed to create a personalized chatbot.

<img src="https://github.com/YanSte/NLP-LLM-Fine-tuning-T5-Small-Reviews/blob/main/img_1.png?raw=true" alt="Learning" width="50%">

### Prompt Datasets

The utilization of chat prompts during the fine-tuning of a T5 model holds crucial significance due to several inherent advantages associated with the conversational nature of such data. Here is a more detailed explanation of using chat prompts in this context:

1. **Simulation of Human Interaction:** Chat prompts enable the simulation of human interactions, mirroring the dynamics of a real conversation. This approach facilitates the model's learning to generate responses that reflect the fluidity and coherence inherent in human exchanges.

2. **Contextual Awareness:** Chat prompts are essential for capturing contextual nuances in conversations. Each preceding turn of speech influences the understanding and generation of responses. The use of these prompts allows the model to grasp contextual subtleties and adjust its responses accordingly.

3. **Adaptation to Specific Language:** By incorporating chat prompts during fine-tuning, the model can adapt to specific languages, unique conversational styles, and even idiosyncratic expressions. This enhances the model's proficiency in generating responses that align with the particular expectations of end-users.

4. **Diversity in Examples:** Conversations inherently exhibit diversity, characterized by a variety of expressions, tones, and linguistic structures. Chat prompts inject this diversity into the training process, endowing the model with the ability to handle real-world scenarios and adapt to the richness of human interactions.

In summary, the use of chat prompts during the fine-tuning of a T5 model represents a potent strategy to enhance its capability in understanding and generating conversational texts. These prompts act as a bridge between training data and real-life situations, thereby strengthening the model's performance in applications such as chatbot response generation, virtual assistant systems, and other natural language processing tasks.

## Learning Objectives

By the end of this notebook, you will gain expertise in the following areas:

1. Learn how to effectively prepare datasets for training.
2. Understand the process of fine-tuning the T5 model manually, without relying on the Trainer module.
3. Explore the usage of accelerators to optimize model training and inference.
4. Evaluate the performance of your model using metrics such as Rouge scores.
