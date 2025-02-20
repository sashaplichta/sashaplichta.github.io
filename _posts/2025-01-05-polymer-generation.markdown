---
layout: default
modal-id: 101
date: 2025-01-05
img: labelled_polymer.png
alt: image-alt
project-date: September 2024 - Present
where: Dr. Yadav's Lab, UBC
category: AI/ML, CHBE
subheadings:
    - Motivation
    - PolyGPT
    - PolyTAO
    - PolyVAE
    - PolyLDM
description:
    Motivation: |
        Polymers are a critical part of day-to-day life, used in everything from plastics to clothes. As we seek to reduce our reliance on petroleum-derived products, the source of many modern polymers, there is an increasing push to develop new polymers that use organic molecules as a feedstock. The search space of possible polymers is too massive to ever explore experimentally. So, we need to extract valuable insights from a massive space governed by complex relationships - seems like a perfect use case for AI. The problem at hand is generative, making it a little more complex. The post below is an up-to-date summary of our approach. You can find the most up-to-date code in the[ GitHub repository](https://github.com/sashaplichta/polymer_models) for this project.

        There are many architectures that have shown state-of-the-art performance on polymer generation tasks. Parallels between the task of generating a polymer sequence using target properties as conditioning and well-researched applications like guided image generation, NLP sequence generation, and protein/molecular design. Some architectures, including conditional Variational Autoencoders (cVAE) and transformer-based encoder/decoders, have been applied very successfully to the problem of de-novo polymer design. Others, like latent diffusion models, have not yet been explored in this specific context. Accordingly, before investing lots of time in a single model/architecture, we want to explore the performance of minimal implementations of several models on a simplified task. In our case, this simplified task is generating molecules based on a battery of electrical properties compiled in the [QM7b dataset](https://moleculenet.org/datasets-1). While not necessarily much easier, we can use a curated subset of this dataset to explore an "ideal" case, avoiding polymer datasets with lots of missing values. The models we want to explore are:
        - A transformer model based on the GPT architecture (PolyGPT), pretrained on a seq-to-seq molecular autoencoding task and then fine-tuned to generate molecules from properties
        - A toned-down implementation of the SOTA polymer-generation model [PolyTAO](https://www.nature.com/articles/s41524-024-01466-5#Sec10)
        - A cVAE model loosely based on [this](https://link.springer.com/article/10.1186/s13321-018-0286-7) molecular-generation model, but using transformer blocks instead of LSTMs for the recursive generation (PolyVAE)
        - A latent diffusion model operating in the same latent space as the cVAE, so that we can use the cVAE decoder to generate the final molecule's SMILES representation (PolyLDM)

        In order to compare the performance of these models, we devised a battery of criteria including: Chemical Validity, Uniqueness (is the generated polymer not in the training set), and Correlation with Target Properties. To evaluate the properties of polymers not present in the dataset, a small(ish) property-prediction model is trained. The model descriptions below assume the model is configured for the ultimate polymer generation task, not the initial electronic property evaluation task.
    PolyGPT: |
        The model described below is an autoregressive generator of polymers based using desired properties as input. Training consists of two stages: pre-training on chemical molecules without property direction, and training on polymers using properties to guide the generation process.

        # Model Design
        ### Architecture
        In order to stabilize training, auxiliary heads are used to predict the "true" properties of the generated molecule. Additional prediction heads should provide more nuanced gradients to guide model training. The main head of the model predicts the next token given the input sequence. The "active" part of the model architecture changes slightly between pretraining and fine-tuning. 

        During **pretraining**, the auxiliary heads do not contribute to the loss, and an embedding of 0s is fed in the place of target properties. The model is trained to generate the example molecules starting with after a \<start> token and terminating in an \<end> token. 

        During **fine-tuning**, a simple MLP is used to encode the target properties into an embedding (the same size as the token embeddings). This is prepended to the \<start> token embedding before inference. During training, a molecule's known properties are slightly perturbed with the perturbation serving as the auxiliary head target. In theory, this should train the model to output how close the molecules true properties are to the desired, target properties.

        ![PolyGPT](img/portfolio/PolyGPT.png)

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

        $$\mathcal{L}_{aux} = \frac1n \sum_j^n (p_j - \hat{p_j})^2$$

        Input Alignment Loss (MSE)

        $$\mathcal{L}_{loss} = \frac1n \sum_j^n (P_j - \hat{P_j})^2$$

        ### Inference
        In order to generate a polymer conditioned on the desired properties, the properties are embedded, then pre-pended to the start token's embedding. This is then iteratively fed into the "body" of the model, with each predicted token appended to the sequence and fed back into the model. Generation ends when the model either predicts the <EOS> token, or runs into the limit on token generation (set both so that the context length isn't exceeded, and so that we don't generate tokens forever).

        ### Data Processing and Tokenization
        Molecules are represented using the SMILES format. For fine-tuning, property targets are normalized according to the training dataset. In the future, missing properties may be imputed using dedicated property models.

        Polymers have some elements and tokens distinct from most drug-like molecules (that form the foundation of many available datasets). In particular, metals may be present and * is used in some datasets to indicate polymerization points. A tokenizer is trained using the fine-tuning dataset of experimentally-validated polymers. Any molecules containing tokens not in the vocabulary are removed from training, though an <UNK> token is included in the model's vocabulary to be safe.

        Rather than use an atomic vocabulary, tokenization is performed by sequentially merging the two most common adjacent tokens until a chosen vocabulary size is achieved. The size of the vocabulary is the subject of experimentation due to the low initial complexity of the dataset (only a few elements are represented), but rapidly increasing complexity when considering larger molecular structures.

    PolyTAO: |
        The PolyTAO model was developed for the purpose of on-demand polymer design from desired properties. It's fundamentally a transformer-based encoder/decoder architecture, based largely off of Google's T5 language models. While it still maps from sequence to sequence, it maps from properties to polymers rather than the molecule-to-molecule architecture that is more common. In order to tokenize the input, the authors map each continuous property to the nearest integer and use these as the basis for the property-encoding vocabulary. The model achieved impressive performance on both chemical validity (99.27% of generations) and correlation between generated polymer properties and their expected properties ($$R^2=0.96$$).

        # Model Design
        ### Architecture
        The model consists of an encoder and decoder, each consisting of 12 transformer blocks. The encoder takes in the property embeddings and maps them to a hidden space. This hidden representation is attended to during the decoding process using cross attention. The decoder takes as input the sequence of token embeddings representing the extent of the polymer generated so far and predicts the next token in the sequence. This direct mapping of polymer properties to sequence emphasizes the relationship between the two during training, improving the model's performance, but requires a well-annotated dataset (something often missing for polymer properties). To overcome this, the authors trained a property-prediction model and used this to approximate the properties of the polymers in PI1M, a large dataset of 1 million polymer structures lacking the corresponding properties. A rough overview of the architecture is below:
        - The properties are first embedded using an embedding lookup
        - Learnable position embeddings are obtained for the property sequence
        - The property embedding sequence and the position embedding sequence are summed and fed to the encoder transformer block
        - Before the decoding process start, the embeddings of the tokens fed into the decoder (the existing sequence) are obtained using another embedding lookup, along with their corresponding position embeddings
        - The token embeddings and position embeddings are again summed and fed into the transformer block. This time, however, the transformer also attends to the encoder's output as it tries to predict the next token
        - A linear layer maps the transformer output to the vocabulary, generating the final logits that will be passed into a softmax function during inference

        ![PolyTAO](img/portfolio/PolyTAO.png)

        ### Training Objective
        The model was trained to generate a polymer given its properties, so the loss for each generated token during training was determined using cross entropy. This is a pretty vanilla loss function as far as autoregressive models go, but I've included the formula to calculate it below for reference:

        $$\mathcal{L}_{token} = - \sum^N_{i=1} y_i \ln p_i$$

        ### Inference
        Since the model is trained to map directly from properties to polymers, inference is very similar to/the same as the training procedure. Instead of known properties, however, the desired properties are fed into the model, encoded, and then fed into the decoder with the growing sequence, starting inference with the sequence [<SOS>, <MASK>]. After each pass through the decoder, the next token is sampled from the predicted distribution, appended to the sequence, and fed back into the model until the termination condition is reached.

        ### Data Processing
        The only really interesting data-related feature of this model is the fact that the input properties are all treated as discrete tokens. This surprised me when I first read through the model implementation - usually continuous properties are retained as continuous properties, since treating them as discrete tokens strips a lot of the semantic information related to their order. For example, while it might be clear to us that a polymer with a glass transition temperature of 150 C and one with 152 C are likely pretty similar, the model does not start out with this inductive bias and must learn this relationship as it learns each token's respective embedding. This was one design decision I wanted to challenge when designing PolyGPT - there, I opted to use a linear layer to map the continuous properties directly to the embedding space in an attempt to preserve these relationships.

    PolyVAE: |
        Variational autoencoders work by using a neural network to predict the mean and log variance of a probabilistic latent space. A random sample is then taken from this latent space and fed to the decoder, which rebuilds the input. Direct sampling from a learned probability distribution, however, disrupts the ability to compute gradients during backpropagation. Instead, we can get around this by taking a random sample from a normal distribution and reparametrizing it using the mean and log variance predicted by the encoder, allowing us to backpropagate through the middle of the model.

        # Model Design
        ### Architecture
        The cVAE architecture implemented in this project is pretty standard. Our model relies on transformer blocks to encode the SMILES input of the polymer into a single embedding that is then fed into the "autoencoder" part of the model. The output of the decoder is then expanded to a SMILES output using transformer blocks with (causal) self-attention. A more detailed breakdown is as follows:
        - Assume we have polymer input (SMILES) Poly and desired properties Prop
        - Poly is mapped to the embedding space using a lookup table to get $$E_{poly}$$ while the continuous properties are mapped to the latent space using a single linear layer, giving $$E_{prop}$$
        - $$E_{prop}$$ is expanded to match the sequence length of $$E_{poly}$$, after which the two are concatenated. The joined tensor is then passed through the encoder_transformer, comprised of 4 transformer layers
        - A linear layer is computed using the output of the transformer block across the embedding dimension, which we then take the softmax of to get an attention mask
        - We then take the element-wise product of the attention mask and the transformer output, before computing the sum along the sequence dimension to compress the transformer output to (batch, embedding)
        - Now that we have a single embedding representing our polymer and properties, we can feed this into the main VAE encoder (a series of linear layers compressing the embedding to the latent dimension)
        - Two linear layers are then used to predict the parameters of the probabilistic latent space (mean and log variance)
        - We then use the reparameterization trick ($$z = \mu + \epsilon * \exp(0.5 * log_var))$$ to get a random sample in a differentiable way
        - The latent sample is concatenated with the latent embedding of the properties, $$E_{prop}$$, and fed into the main VAE decoder (another series of linear layers expanding from the concatenated latent space to the embedding space)
        - The decoded z is then prepended to the token embeddings of the input sequence ([<SOS>, <Mask>] to start) 
        - A transformer block with causal self-attention is then used to autoregressively generate the polymer output by predicting the embedding of the <Mask> token, which is then fed to a linear layer mapping it to the vocabulary

        ![PolyVAE](img/portfolio/PolyVAE.png)

        ### Training Objective
        During training, the loss function is defined by two components: reconstruction error, and Kullback-Leiber (KL) divergence. The reconstruction error, as in the transformer-based models, is a measure of how closely the model output resembles the input molecule. The KL divergence, however, represents a measure of how closely the VAE's learned parameters adhere to the normal distribution. This represents a form of regularization and helps keep the model from overfitting, something that will be important when the model is used for polymer generation. The loss functions are summarized below:

        Token Loss (Cross-Entropy Loss)

        $$\mathcal{L}_{token} = - \sum^N_{i=1} y_i \ln p_i$$

        KL Divergence

        $$D_{KL}(P||Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}$$

        ### Inference
        During inference, desired polymer properties are fed into the model and embedded. A random sample is then drawn from the latent space and concatenated with the embedded polyemer properties before being decoded as usual. The use of a random sample enables the generation of numerous polymers for a given set of polymer properties, as well as enables (relatively) easy exploration of the VAE's learned distribution.

        ### Data Processing
        The VAE's handling of data is a little messy since it needs to map an arbitrary sequence length to a single embedding vector, model the latent distribution, and map back to a sequence, all while considering the property conditioning. I'm honestly not sure how the best way to incorporate this conditioning is so I imagine this model may change a bit. In particular, we could add the conditioning vector before the first transformer block, encorporating it into the embedded representation, or we could incorporate it after we derive the polymer embedding. It'll probably take some tuning to figure out what works best, so I'll be sure to update this as we try new things.

    PolyLDM: |
        I have not (so far) found an example of anyone using a latent diffusion model for polymer design. I have, however, found some interesting papers describing how they can be used for molecular/drug design, so I wanted to see if they could be adapted to our problem here. Latent diffusion models are perhaps most well known for their incredible performance on image generation tasks (think Stable Diffusion), but they have found wide application with other generative problems. Latent diffusion models, like all diffusion models,  learn to progressively de-noise the input. This allows them to generate the output from pure noise, enabling the generation of diverse outputs very easily, much like VAEs. Unfortunately, denoising a large input can get very expensive (images or videos might be the most intuitive example). What's interesting about latent diffusion models in particular is the fact that they bypass this limitation by performing the denoising process in latent space before passing the denoised representation to a decoder. In our case, the latent diffusion model operates in the same latent space the cVAE model and uses its decoder to generate the final output. This means that their performance is intertwined, and if we train a bad cVAE model it'll be hard/impossible to get a good latent diffusion model, making it a little challenging to explore these models in a bakeoff like this.

        # Model Design
        ### Architecture
        Latent diffusion model architectures can be quite diverse. Ordinary diffusion models often favor a U-Net architecture due to its ability to capture both high-level and granular features. In our case, however, the input is already in a compressed latent space. In theory, this means we could get away with just a standard MLP network with a bunch of same-sized layers. While I do explore this, I wanted to also test if we could do better by adapting something more like a U-Net, but without any convolutions. So, we've got two (small) models to test in this section:

        **Simple MLP**
        - The desired properties are embedded using the cVAE's learned property embedder (this is just to initialize the weights - the embedding layer is unfrozen part way through training)
        - The noise input is concatenated the property embedding
        - The first layer maps the concatenated vector from the latent space (32) to 256
        - We pass the output of the first layer through 2 more layers of size 256
        - Finally we contract from 256 back to the latent dimension

        **Adapted U-Net**
        - The desired properties are embedded using the cVAE's learned property embedder (this is just to initialize the weights - the embedding layer is unfrozen part way through training)
        - The noise input is concatenated the property embedding
        - In our first layer, we expand the concatenated vector from the latent space (32) to 256
        - Next we contract to 2 * the latent space
        - We then map from 2 * the latent space back to 256, before concatenating the output of this layer with the output of the first layer
        - Finally, we map back from 256 to the latent space, obtaining the final, denoised vector

        A rough overview of the Adapted U-Net architecture is shown below.

        <img src="img/portfolio/PolyLDM.png" width="600">

        ### Training Objective
        Diffusion models work by removing noise from the input, so our training loss function is effectively just the reconstruction error between our model's output and the previous noise step (so one step before the input). This can be calculated for each time step (i) using the formula below:

        $$ Loss_i =  \mathbb{E}_{z,e,t} \left[ || \epsilon - \epsilon_{\theta} (z_t, t) ||^2_2 \right]$$

        ### Inference
        During inference, the desired properties are fed into the model along with a randomly sampled vector of pure noise. Over successive iterations, the noisy vector is progressively denoised until the final "time" step, at which point it is fed into the cVAE's decoder to obtain the final polymer sequence.

        ### Data Processing
        This model is pretty straightforward. One thing worth noting, since at time of writing I haven't actually trained these models yet, is that I'm unsure how the fact that we are training a conditional VAE, rather than an unconditioned VAE, will affect our model. In essence, the output of the latent diffusion model gets conditioned on the input twice, once during the diffusion process, and again during the decoding process. This could mean that the outputs of this model will outperform the others, but I'm that feeding the property vector in twice like this could lead to some sort of model collapse.
---