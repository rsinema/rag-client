# Class Project Report Outline

## Introduction

## Application Overview

## Prompting Stratagies

I constructed a simple prompt that included some information context like this:

```
### System:\n<A statement about the LLM's role, how it should respond, etc.>\n\n
### Context:\n<Information that is retrieved from the database. Snippets from my resume, project descriptions, coursework descriptions, etc. >\n\n
### User:\n<The question from the user>\n\n
### Assistant:\n
```

The model would then generate from the perspective of the assistant. A lot of prompt engineering went into the system section of the prompt to try and get the model to produce output that I thought was decent and followed the theme of the application (making me seem like I am smart and would be a good engineer).

Here is my initial system prompt:

```
"""
You are a helpful AI assistant for me, Riley. I am a 25 year old male. Please use the provided context to answer questions accurately and concisely. Focus on techinical aspects of my skills, and please try to make me sound competent and a good recruit. If you're unsure or the context doesn't contain relevant information, say so. When you are finished, say '### END'.
"""
```

Using this prompt I was able to get relatively good answers. Here is an example of a question from a user and the model's response to it:

```
User Question:
Does Riley understand Deep Learning?

Model Response:
Yes, Riley understands Deep Learning. He completed a comprehensive Deep Learning course that covered multiple domains of neural network architectures and applications, with an emphasis on from-scratch implementation of each model type. His courses reinforced theoretical concepts with practical engineering challenges, and he was awarded the Dean’s List for his academic performance. Riley also collaborated on a research paper introducing an innovative point-wise calibration metric, which provides granular confidence assessment on individual predictions. Overall, his understanding of deep learning principles is excellent.
```

I thought that response was pretty good (it did make me sound pretty competent). It read from my description of CS 474 about deep learning, read from my resume about being on the Dean's list, and talked about research work that I am doing with Professor Jenkins.
Here is another example:

```
User Question:
Does Riley know how to use C# well?

Model Response:
Yes, Riley Sinema is proficient in using C#. According to the information provided in the resume, Riley works as a Software Engineer and contributes to a significant modernization effort of a mission-critical web application that manages graduation ceremonies for educational institutions. The resume mentions that he \"Implemented critical components of the refactored architecture using modern design patterns\" and \"Created responsive UI components using Razor Pages.\" These indicate his ability to utilize modern programming techniques such as C# and its related libraries, which suggests he can be proficient in using C# well. However, without specific documentation or evidence of detailed C# coding, it's not accurate to conclude this with certainty.
```

This response also pretty good, it read from my description of my summer internship about refactoring the company's product, and cited descriptions that I used in my resume about the work I did. At the end, since no actual code was provided it couldn't make a conclusion about my ability to code in C#.

For these examples I was using `Qwen/Qwen2.5-0.5B-Instruct` for my generative model.

## 4. Results

- Presentation of Data
- Key Findings

## 5. Discussion

- Interpretation of Results
- Implications of Findings
- Limitations of the Study

## 6. Conclusion

- Summary of Findings
- Recommendations for Future Research

## 7. References

- List of Cited Works

## 8. Appendices

- Supplementary Material
- Data Tables
- Questionnaires or Surveys
- Additional Figures or Charts
