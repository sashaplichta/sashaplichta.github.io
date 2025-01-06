---
layout: default
modal-id: 7
date: 2025-01-04
img: labelled_thesis_redux.png
alt: image-alt
project-date: January 2025 - Present
where: Dr. Frostad's Lab, UBC
category: AI/ML, CHBE
description: |
    Building on the model and algorithm I developed as part of my undergraduate thesis, I've been working to integrate the two solving approaches. In general, I found the model was good at making predictions that matched the continuity and structure expected of a real film, but struggled to be bound to the actual film. In contrast, the color-matching algorithm excelled at matching the film, particularly as more reference colors were used, but sometimes predicted aberrant points far off the surface. Here, I explore two corrections. The first enforces continuity on the color-matching algorithm, and the second involves training a new model to predict the correct surface structure from the color-matching algorithms output. In a departure from my previous work, all solving now happens in a normalized color space that should, hopefully, be agnostic to the components (and their refractive indices) of the system, so long as they are known.

    # Color Normalization

    # Continuity Enforcement

    # Hybrid Model
---
