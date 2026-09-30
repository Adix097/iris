# Known Limitations

## Lens ROI accuracy when viewed from oblique angles
If the person turns their head to a significant degree relative to the camera,
the estimated lens ellipse may diverge from the true lens location, especially
for the farther eye. This happens since the fitting of facial landmarks (LBF,
68-points) is inaccurate in case of partial self-occlusion of eye corners,
and hence the geometry of the lens based on these points becomes inaccurate.

Works well in: head facing the camera or slight turns (typical on call
position). Fails at: steep head angle / near-profile.
