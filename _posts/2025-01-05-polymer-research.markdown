---
layout: default
modal-id: 1
date: 2025-01-05
img: cabin.png
alt: image-alt
project-date: Ongoing
where: Dr. Yadav's Lab, UBC
category: AI/ML, CHBE
description: |
    Polymers are a critical part of day-to-day life, used in everything from plastics to clothes. As we seek to reduce our reliance on petroleum-derived products, the source of many modern polymers, there is an increasing push to develop new polymers that use organic molecules as a feedstock. The search space of possible polymers is too massive to ever explore experimentally. So, we need to extract valuable insights from a massive space governed by complex relationships - seems like a perfect use case for AI. The problem at hand is generative, making it a little more complex. The post below is an up-to-date summary of our approach. As changes are made, I'll change the main text below and add a comment to the bottom describing the update.

    ## Polymer GPT Model Workflow
    The model described below is an autoregressive generator of polymers based using desired properties as input. Training consists of two stages: pre-training on chemical molecules without property direction, and training on polymers using properties to guide the generation process.

    # Model Design
    ### Architecture
    In order to stabilize training, auxiliary heads are be used to predict the difference between the desired and \"true\" properties of the generated molecule. Additional prediction heads should provide more nuanced gradients to guide model training. The main head of the model predicts the next token given the input sequence. The model architecture changes slightly between pretraining and fine-tuning. 

    During **pretraining**, the auxiliary heads do not contribute to the loss, and a 0 embedding is fed in the place of target properties. The model is trained to generate the example molecules starting with after a \<start> token and terminating in an \<end> token. 

    During **fine-tuning**, a simple MLP is used to encode the target properties into an embedding (the same size as the token embeddings). This is prepended to the \<start> token before inference. During training, a molecule's known properties are used the target.

    ### Objective Function (Pretraining)
    During pretraining, the model is evaluated on its ability to predict the correct next token with the objective function, and whether the final molecule is chemically valid: 
    $$\mathcal{L}_{pt} = \alpha \mathcal{L}_{token} + \beta \mathcal{L}_{valid}$$

    Token Loss (Cross-Entropy Loss)
    $$\mathcal{L}_{token} = - \sum^N_{i=1} y_i \ln p_i$$

    Validity Loss (binary yes/no from *Chem.MolFromSmiles('SMILES_formula')*)
    $$\mathcal{L}_{valid} = 0, 1$$

    ### Objective Function (Fine-tuning)
    During fine-tuning, the model is evaluated on its ability to predict the correct next token and the validity of the generated molecule, as well as how closely the properties of the final molecule align with the input properties and how well each of the auxiliary head properties are predicted. Each of these objectives is weights according to the parameters $\alpha$, $\beta$, $\gamma$, and $\delta$ to produce the objective function:
    $$\mathcal{L}_{ft} = \alpha \mathcal{L}_{token} + \beta \mathcal{L}_{valid} + \gamma \mathcal{L}_{aux} + \delta \mathcal{L}_{align} $$

    Token Loss (Cross-Entropy Loss)
    $$\mathcal{L}_{token} = - \sum^N_{i=1} y_i \ln p_i$$

    Validity Loss (binary yes/no from *Chem.MolFromSmiles()*)
    $$\mathcal{L}_{valid} = 0, 1$$

    Auxiliary Head Loss (MSE)
    $$\mathcal{L}_{aux} = \frac1n \sum_j^n (p_j - \^p_j)^2$$

    Input Alignment Loss (MSE)
    $$\mathcal{L}_{loss} = \frac1n \sum_j^n (P_j - \^P_j)^2$$


    # Datasets and Data Format
    ### Datasets
    - PoLyInfo (https://polymer.nims.go.jp/datapoint.html): lots of polymer structure/property pairs

    ### Data
    Molecules are represented using the SMILES format. For fine-tuning, property targets are normalized according to XXX. Missing properties are imputed using dedicated property models according to the scheme scheme below.
    - $\lambda$ - 
    - $T_g$ - 
    - XXX

    ### Tokenization
    Polymers have some elements and tokens distinct from most drug-like molecules (that form the foundation of many available datasets). In particular, metals may be present and * is used in some datasets to indicate polymerization points. Tokenization is performed using the INSERT dataset of experimentally-validated polymers. Any molecules containing tokens not in the vocabulary are removed from training.

    Rather than use an atomic vocabulary, tokenization is performed by sequentially merging the two most common adjacent tokens until a vocabulary size of INSERT is achieved. Tokenization is performed using the DeepChem SMILESTokenizer.

    # Pretraining


    # Fine-tuning
---
