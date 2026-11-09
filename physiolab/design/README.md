# Imported design reference

Source: user's `PhysioLab Builder Import.zip`, received 2026-11-08. Kept the top-level
HTML, runtime, design system, screenshots, XR page and image assets. The nested duplicate
`uploads/PhysioLab builder interface/` and archive thumbnail were omitted.

Serve with `python3 -m http.server 8098 --directory design` from `physiolab/`.
Open `/PhysioLab%20Builder.dc.html`. This is the original interactive design prototype;
network access is needed for its React/icon/font/XR CDN dependencies. The working application
is in `web/` and uses the same design tokens. Mock device/model/deploy statuses in this
reference are not evidence of real integrations.
