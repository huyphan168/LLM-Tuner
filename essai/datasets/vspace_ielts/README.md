# Vspace IELTS Essay Rewriting Dataset

## Dataset cards for different instructions
Dataset             | Use Case
---------------------------------------     | ---------------------------------------------------
rewritten essays               | Given a topic and a student essay, rewritten the essay with higher quality
coherence                      | Given a topic and a student essay, give coherence summary of the essay
vocab                          | Given a topic and a student essay, give vocabulary review of the essay and give alternative words

## Dataset Description

This data was created by providing rewritten version of draft IELTS essays of the students.

Example:

```
User: Given following ielts topic: Today more people are overweight than ever before. Some believe it is the government’s responsibility to tackle the situation, while others argue for individual self-care.
Discuss both views and give your own opinions and student essay: Due to the ubiquity of sweetened beverages and cheap fast food chains, the globe has to face a big fat crisis - a debated issue whether it is to profoundly blame the government or individual self-management. In this essay, both views will be cited with reliable evidence. Rewrite the essay with higher quality


Lehrer: Due to the ubiquity of sweetened beverages and processed food, the globe potentially has to face an obesity pandemic. Some people claim that the government should take responsibility for this issue, while others argue that individual self-management is profoundly to blame. From my perspective, both sides, which will be discussed in this essay, are of duty to tackle this alarming situation 

```

# Usage

The dataset contains one configuration, `text-generation`, which has a single
text `text-generation` feature.

# Source data
