---
layout: default
modal-id: 104
date: 2024-08-01
img: labelled_bioreactor.png
alt: image-alt
project-date: September, 2024 - April, 2025
where: UBC
category: CHBE
description: |
    As part of our capstone course, I developed a model to describe the availability of nutrients throughout the fermentation process during the production of a heterologous molecule in the yeast *S. cerevisiae*. This isn't meant to be a comprehensive write-up; it's more of a description of our thought process. The actual code can be found in [this repository](https://github.com/sashaplichta/CHBE-481). The model describes three phases of growth commonly exploited in industrial processes and takes into account biomass concentration, glucose concentration, oxygen uptake/transfer rates, and product concentration throughout the fermentation. The product we're modelling is produced constitutively, so I don't consider any inducer.

    # Background
    In industrial fermentation (specifically in pharma) there are a number of organisms that can be used to produce a heterologous (non-native) product. Typically, however, large bodies of research, tools, and precedence (important for the FDA) exist for a small subset of possible organisms: yeasts (*P. pastoris*, *S. cerevisiae*, and some others), bacteria (*E. coli*, *B. subtilis*), mammals (CHO), and some insects. Organisms are selected on the basis of many factors, but the complexity and nature of the product, as well as what species the underlying research was conducted in, can play an outsized role. In our case, we're producing a molecule that's shown promise as a treatment for Type II Diabetes. Since the original research was conducted in *S. cerevisiae* and it's a commonly used and well characterized host, that's what we'll be modelling.
    
    # Fermentation Stage I: Batch
    At the start of our fermentation, we innoculate a large (20,000 L) bioreactor with 0.01 g/L of cells. To prevent overflow, our working volume is actually 16,000 L, or 80% of our total volume. Initially, our glucose concentration is set high (5 g/L) to encourage rapid growth. Before starting any recycle systems or feeds, we let our cells grow untouched until the glucose concentration falls to some set floor. This floor is one of the main parameters we can tweak to control the rate of cell growth, which is governed by the Monod equation below. Critically, the oxygen demands of our culture depend on the growth rate and cannot exceed the physical limitations of our system.

    $$ \mu_g = \frac{\mu_{max} S}{K_S + S} $$

    In the Monod equation, $$K_S$$ is a property of the species in question and doesn't change. The actual culture growth is governed by two relatively simple ODEs:

    $$ \frac{dX}{dt} = \mu_g X $$

    $$ \frac{dS}{dt} = - \frac{\mu_g}{Y_{X/S}} X - m_S X $$

    Here, $$Y_{X/S}$$ and $$m_S$$ are two coefficients that are again, characteristic of the strain being used. $$Y_{X/S}$$ is the yield of cell mass on substrate mass (derived from a chemical model of *S. cerevisiae*). Since we know the initial conditions, we can iteratively solve these equations until our concentration of glucose hits our floor. Initially, we set a substrate floor of 0.05 g/L.

    # Fermentation Stage II: Fed-Batch
    Once our culture has reached the substrate floor, we switch to fed-batch operation. All this means is that we provide a concentrated glucose feed to maintain the substrate concentration in the culture at the floor. At the same time, we remove the same volume from the culture to keep our overall volume constant. We now have three (really two, as you'll see) coupled ODEs describing our culture and its limitations:

    $$ \frac{dS}{dt} = 0 $$

    $$ \frac{dX}{dt}=\frac{F}{V}\left(\alpha C_XX-\left(1+\alpha \right)X\right)+\mu _gX $$

    $$ F = \frac{V}{S_{feed} + \alpha C_S S - (1 + \alpha S)} (m_S X + \mu_g \frac{X}{Y_{X/S}}) $$
    
    $$ OUR=q_{O_2}\frac{\mu _g}{\mu _m}X+m_sXY_{\frac{O_2}{S}} $$

    Now, our substrate concentration is not changing, but our cell concentration in the reactor is continuously diluted or "flushed" out. The strength of this dilution depends on both the feed flow rate (F) and the cell recycle ratio ($$\alpha$$). Finally, our culture is now limited by the oxygen transfer rate (OTR) of our system. Thus, we need to monitor the oxygen uptake rate (OUR) until it reaches our oxygen transfer limit. The OUR is a function of the growth rate and the cell maintenance. This phase continues until our OUR is equal to our OTR, at which point we need to start getting creative.

    # Fermentation Stage III: Modified Fed-Batch
    Our culture is now consuming as much oxygen as we provide it, so we need to decrease the growth rate as our cell mass increases to maintain that relationship. This is a relatively delicate procedure, and to my understanding does not often occur in industry, as typically you would now induce and shift your culture from being growth-oriented to being production-oriented. In our case, however, our product is coupled to growth, so we need to maximize our cell density. Now, our culture is defined by the following equations. Notice, despite the fact that these are written as ODEs, they are solved iteratively in time steps of 0.001 hours. Before, we could solve our systems of ODEs using tools like SciPy's odeint.

    $$ OUR=q_{O_2}\frac{\mu _g}{\mu _m}(X + dX) + m_s(X + dX) Y_{\frac{O_2}{S}} $$

    $$ \mu_g = \frac{\mu_{max} S}{K_S + S} $$

    $$ F = ((S_t - S_{t-1}) + m_S X + \frac{\mu_g}{Y_{X/S}} X) \frac{V}{S_{feed} + \alpha C_S S_t - (1 + \alpha) S_t} $$

    $$ \frac{dX}{dt}=\frac{F}{V}\left(\alpha C_XX-\left(1+\alpha \right)X\right)+\mu _gX $$

    At each time step, we need to solve for what growth rate keeps our OUR = OTR, then solve for the substrate concentration necessary for this growth rate. Once we know the substrate concentration, we can solve for the feed flow rate (which is the same as the rate of culture loss). Finally, we can solve for the change in cell mass over the time step. We continue this until the time reaches 36 hours, our self-appointed max fermentation time. You could, however, extend this time to find the "carrying capacity" of the bioreactor, at which point the OUR is solely due to maintenance and is equal to the OTR.

    # Results
    Check out the repository linked above for the complete code, but as a snapshot, the graph below shows the concentration of biomass and glucose over the course of the fermentation described above.

    ![Graphical Results](img/portfolio/fermentation_results.png)


---