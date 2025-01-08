---
layout: default
modal-id: 101
date: 2025-01-05
img: labelled_polymer.png
alt: image-alt
project-date: September 2024 - Present
where: Dr. Yadav's Lab, UBC
category: AI/ML, CHBE
description: |
    Polymers are a critical part of day-to-day life, used in everything from plastics to clothes. As we seek to reduce our reliance on petroleum-derived products, the source of many modern polymers, there is an increasing push to develop new polymers that use organic molecules as a feedstock. The search space of possible polymers is too massive to ever explore experimentally. So, we need to extract valuable insights from a massive space governed by complex relationships - seems like a perfect use case for AI. The problem at hand is generative, making it a little more complex. The post below is an up-to-date summary of our approach. As changes are made, I'll change the main text below and add a comment to the bottom describing the update. You can find the most up-to-date code in the[ GitHub repository](https://github.com/sashaplichta/polymer_models) for this project.

    ## Polymer GPT Model Workflow
    The model described below is an autoregressive generator of polymers based using desired properties as input. Training consists of two stages: pre-training on chemical molecules without property direction, and training on polymers using properties to guide the generation process.

    # Model Design
    ### Architecture
    In order to stabilize training, auxiliary heads are be used to predict the difference between the desired and "true" properties of the generated molecule. Additional prediction heads should provide more nuanced gradients to guide model training. The main head of the model predicts the next token given the input sequence. The "active" part of the model architecture changes slightly between pretraining and fine-tuning. 

    During **pretraining**, the auxiliary heads do not contribute to the loss, and am embedding of 0s is fed in the place of target properties. The model is trained to generate the example molecules starting with after a \<start> token and terminating in an \<end> token. 

    During **fine-tuning**, a simple MLP is used to encode the target properties into an embedding (the same size as the token embeddings). This is prepended to the \<start> token embedding before inference. During training, a molecule's known properties are slightly perturbed with the perturbation serving as the auxiliary head target. In theory, this should train the model to output how close the molecules true properties are to the desired, target properties.

    ### Objective Function (Pretraining)
    During pretraining, the model is evaluated on its ability to predict the correct next token with the objective function: 

    $$\mathcal{L}_{pt} = \alpha \mathcal{L}_{token}$$

    Token Loss (Cross-Entropy Loss)

    $$\mathcal{L}_{token} = - \sum^N_{i=1} y_i \ln p_i$$

    ### Objective Function (Fine-tuning)
    During fine-tuning, the model is evaluated on its ability to predict the correct next token, as well as how closely the properties of the final molecule align with the input properties and how well each of the auxiliary head properties are predicted. Each of these objectives is weighted according to the parameters $$\alpha$$, $$\beta$$, and $$\gamma$$ to produce the objective function:

    $$\mathcal{L}_{ft} = \alpha \mathcal{L}_{token} + \beta \mathcal{L}_{aux} + \gamma \mathcal{L}_{align} $$

    Token Loss (Cross-Entropy Loss)

    $$\mathcal{L}_{token} = - \sum^N_{i=1} y_i \ln p_i$$

    Auxiliary Head Loss (MSE)

    $$\mathcal{L}_{aux} = \frac1n \sum_j^n (p_j - \^p_j)^2$$

    Input Alignment Loss (MSE)

    $$\mathcal{L}_{loss} = \frac1n \sum_j^n (P_j - \^P_j)^2$$


    # Datasets and Data Format
    ### Datasets
    - TBD

    ### Data
    Molecules are represented using the SMILES format. For fine-tuning, property targets are normalized according to the training dataset. In the future, missing properties may be imputed using dedicated property models.

    ### Tokenization
    Polymers have some elements and tokens distinct from most drug-like molecules (that form the foundation of many available datasets). In particular, metals may be present and * is used in some datasets to indicate polymerization points. A tokenizer is trained using the fine-tuning dataset of experimentally-validated polymers. Any molecules containing tokens not in the vocabulary are removed from training, though an <UNK> token is included in the model's vocabulary to be safe.

    Rather than use an atomic vocabulary, tokenization is performed by sequentially merging the two most common adjacent tokens until a chosen vocabulary size is achieved. The size of the vocabulary is the subject of experimentation due to the low initial complexity of the dataset (only a few elements are represented), but rapidly increasing complexity when considering larger molecular structures.

    # Initial Testing
    Many existing polymer datasets are not easily available, so to validate the model implementation, I trained it on this Kaggle dataset for drug discovery. Only two properties are considered (SAS and qed). Initial experiments suggest that the implementation is correct, but a more difficult test needs to be conducted to confirm that the model is learning well.

---
