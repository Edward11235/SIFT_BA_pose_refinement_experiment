# OBA

## Background
This experiment aims to check whether we can use feature point matching and batch adjustmnet together to refine camera pose. Based on result of OBA_test.ipynb, it dose not show promising result. However, below are the two possible direction that could make it works:\
1, Change the optimizer. The currently optimizer is too simple for our pose refinement application. \
2, Use graphic computing algorithm for pose refinement instead of numerical method to compute pose adjustment. By this, I mean that, we can first only leave the matched points that are basically on the same position in picture (since we assume two subsequent images are close in time). Then, the quality of feature point matching will be significantly improved. Then, we use an algorithm to refine 7D pose and move the corresponding matching point to the same place.

## One thing I want to explain
In this algorithm, it maps the 2d point on synthesized image to 3d global frame, and then maps them back to 2d point. The reason I do it is to encode the 7D pose into the computation of loss function. But, I don't think the scipy.optimize.least_squares is doing what I expected it to do.