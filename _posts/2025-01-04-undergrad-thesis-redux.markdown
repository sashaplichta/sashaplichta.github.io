---
layout: default
modal-id: 107
date: 2025-01-04
img: labelled_thesis_redux.png
alt: image-alt
project-date: January 2025 - Present
where: Dr. Frostad's Lab, UBC
category: AI/ML, CHBE
subheadings: False
description: |
    Building on the model and algorithm I developed as part of my undergraduate thesis, I've been working to integrate the two solving approaches. In general, I found the model was good at making predictions that matched the continuity and structure expected of a real film, but struggled to be bound to the actual film. In contrast, the color-matching algorithm excelled at matching the film, particularly as more reference colors were used, but sometimes predicted aberrant points far off the surface. Here, I explore two corrections. The first attempts to smooth the color-matching algorithm, and the second involves normalizing the color space and training a new model to predict the surface structure. The model's solving now happens in a normalized color space, a departure from my previous work that should, hopefully, help it be agnostic to the components (and their refractive indices) of the system, so long as they are known. 

    # Color Normalization
    One of the big challenges with using AI/ML to solve this problem of mapping from a film's interference pattern to a 3D structure is the fact that the color that corresponds to a given thickness changes depending on the refractive indices of the component materials. To get around this, I tried two approaches: integrating the refractive indices into the model as an input, and normalizing the observed colors using a characteristic frequency. This first is relatively self explanatory (though I'll go through it more in the model section below), but the second approach is a little more nuanced. In a color map, like the one shown below, colors repeat after a while. If we call the time from a colors first appearance to its second on "cycle", we should be able to normalize our interference pattern and move it from the color space to a cycle space, right?

    # Continuity Enforcement
    An oversight on my part in my first implementation was my lack of output smoothing. I tried to do some naive smoothing using a convolution to assign each point to the mean or median of its neighbours, but this had mixed results. Now, I've taken a computer vision course, and low and behold, the fourth week covered Gaussian smoothing, high/low pass filtering, and bilinear kernels! So, my revised approach explores each of the following:
    - A high-pass filter to find points that are likely to be aberrations
    - A Gaussian kernel to slightly blurr the output
    - A bilinear kernel to preserve interesting features like terraces and mesas

    The reason I initially leaned towards using a high-pass filter based removal over the two blurring operations is that blurring is, well, blurring. Ideally we wouldn't need to sacrifice resolution in favor of a smooth output. That said, the topology of many systems is naturally smooth (with the exception of some tiny features, as shown in [this paper]()), so blurring may well be a reasonable approach.  

    # Updated ML Model
    Not much is changing off the bat in terms of our model architecture. I'd like to explore training a model whose predictions are conditioned on the refractive indices of the system components, but realistically I know this is likely a crazy inefficient solution given that we can model their interactions mathematically fairly easily. So, instead, I'm focusing on training the model to operate in a normalized color space. Below, you can see the color image corresponding to different film thicknesses, as well as the same image represented in "cycle space". What this allows us to do is have our model predict the number of "cycles" of the representative thickness at each point rather than the actual thickness. Since we know our system, we can then easily map from cycle space back to the 3D space.

    Insert image here

    Our model architecture is a pretty standard U-Net, just like in the first solving attempt. It consists of 5 contracting convolution + pooling layers and 5 expanding + upsampling layers. For each expanding layer, the output from the previous layer is concatenated with the corresponding (same size) contracting layer's output, as in the image below.

---
