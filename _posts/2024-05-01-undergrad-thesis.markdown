---
layout: default
modal-id: 105
date: 2024-05-01
img: labelled_thesis.png
alt: image-alt
project-date: September, 2023 - May, 2024
where: Dr. Frostad's Lab, UBC
category: AI/ML, CHBE
description: |
    This project isn't new - I've just included a description of my undergraduate thesis up to it's completion. An ongoing version of this project can be found under "Undergrad Thesis Redux". Dr. Frostad's lab focuses on studing the physics of thin films. As part of this research, they often need to characterize the structure of composite films from an interferogram - basically a color image of the refraction pattern (think of the rainbow from oil on a puddle). Existing methods for this can be fairly manually intensive as they require annotating a collection of points to generate a descriptive mesh. My thesis focused on developing a way to automate this process, and you can find the resultant code in [this](https://github.com/sashaplichta/thesis/tree/main/data_generation) repository.

    # Background
    When a white light is shown on a film, a variety of colors are observed as a result of constructive and destructive interference. The colors observed depend on the refractive indices of the components of the system, as well as the thickness of the film. Since the observed color is a direct function of the local thickness of each component, the refractive indices of the components, and the composition of the light source, it is possible to easily map from a known system to the observed interferogram. Going the other direction, however, is complicated by the fact that the mapping from system to color is degenerate. That is to say, the observed colors are not unique to a given system or thickness. In an experimental system, interferograms are typically captured as a video of an evolving film as part of a process called dynamic thin film interferometry.

    # My Approach
    In order to keep things simple, I only consider a single system - silicon oil and water. Two existing SOTA methods are used as a basis for assessing performance. The first exploits the fact that for the first 100 nm of thickness, the red, green, and blue wavelengths behave similarly, enabling accurate mapping from color to thickness according to the equation below:

    $$  $$

    Beyond 100 nm, however, the three wavelengths diverge. For thicker films, points can be accurately matched to a theoretical color map to determine the local thickness from observed color. This is the method preferred by Dr. Frostad's lab, and works well for thicknesses up to 10 $$\mu m$$. The color map is generate using equations like those below that relate thickness and intensity at a given wavelength.

    $$  $$

    In order to test my algorithms, I generated synthetic data modelling a silicon oil and water system. To simulate real-world conditions, I added Gaussian noise to the input interferogram to create a noiseless (a) and noisy (b) version for performance comparisons.

    ![Noisifying](img/portfolio/thesis_noisifying.png)
    
    ### Color Matching
    I first explore an algorithm to automate the manual color matching process. In my mind, this works a bit like a k-nearest-neighbors algorithm in the sense that it computes a euclidean distance between the input point and a dataset of points (in our case, the theoretical color map). The color map, however, is continuous. To work in the RGB space, and to simplify computation, it helps to limit ourself to a given number of possible colors. Here, I did use k-nearest-neighbours to find the optimal colors to represent a colormap up to a thickness of 10 $$\mu m$$. You can see the results using 64 colors below.
    
    ![CMap](img/portfolio/thesis_indexing.png)
    
    When the color matching algorithm calculates the distance from itself to each color in the map, it generates a list of candidate colors that "match" our input color to a specified degree. Each of these candidate colors corresponds to a theoretical thickness. To select a "true" thickness from these candidates, we need to use some sort of heuristic. In my case, I decided to use something analogous to the momentum term sometimes used in gradient descent. Here, we pick the thickness closest to the derivative of the previous two thicknesses at that point, extended over the time step between our current frame and the previous. That is, to find a thickness at time $$i$$ ($$t_i$$), we use the thicknesses at the previous two time steps ($$t_{i-1}, t_{i-2}$$) according to the following:

    $$ t_i = \frac{t_{i-1} - t_{i-2}}{(i-1) - (i-2)} ((i) - (i-1)) + t_{i-1} $$

    For this to work, we need to move backwards through our collected video, starting with a fully evolved, or flat, film (0 thickness). I also explore using a points "neighborhood", or the 3x3 grid surrounding it, to predict thickness in order to minimize the propogation of aberrant points through the prediction. In this case, each $$t_{i-x}$$ above would be the average of the neighbourhood rather than a single point's history.
    
    ### Machine Learning (of course)
    Here, I have to thank my supervisor for giving me the latitude to explore what I was interested in. I learned a lot over the course of this work, and in hindsight, machine learning is probably not the best way to solve this problem. That said, I think there's a lot of value in integrating machine learning with an algorithm like the color-matching described above.

    I wanted to explore whether I could train a simple U-Net architecture to predict thickness from an input interferogram. To do so, I normalized the color input image to between 0 and 1 and fed it into a convolutional neural network with 5 "down", or contracting layers, and 5 "up", or expanding layers. The final layer of the model was simply a ReLu layer mapping back to the input image size. I used [this](https://github.com/milesial/Pytorch-UNet) repository as a basis for my model and training. Critically, I only generated/trained on images representing one system - silicon oil and water. While the color-matching algorithm should adapt to other systems, the trained model would likely break as the relationship between thickness and color changes with the refractive indices of the materials - hence why machine learning isn't the best solution approach.

    # Results
    Both approaches ended up performing better than I expected. The color-matching algorithm was more accurate when correct, but had a habit of predicting aberrant thicknesses. In contrast, the machine learning algorithm learned nicely to predict smooth, realistic surfaces, but struggled more when the surface was extreme or noisy, as you can see in the examples below.

    Here, you can see how well the color-matching algorithm (b) predicts the "true" profile (a) when no noise is present. For these examples, a color map with 64 colors was used.

    ![Noiseless](img/portfolio/thesis_kmeans_noiseless.png)

    When noise is added to the input image, however, the performance rapidly decreases. Again, this shows the "true" profile (a) and the prediction (b).

    ![Noisy](img/portfolio/thesis_kmeans_noisy.png)

    Finally, we can see how the machine learning model's prediction (b) better retains smoothness when the input image is noisy, but struggles to match the "true" profile (a).

    ![ML](img/portfolio/thesis_ml_noisy.png)
---
