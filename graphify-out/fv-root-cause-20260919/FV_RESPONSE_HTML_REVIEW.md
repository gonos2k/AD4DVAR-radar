# Original HTML response panel — 2026-09-19

Aside browser inspected the original `examples/weather_scenarios/index.html`.
The new panel uses actual saved `[3,240,240]` sensitivities from
`rotation240_stable_response_18.pt`, sharing one signed colour scale. It separates
the refined response calculation from the original forecast arrays and displays
actual finite-reanalysis changes, linear predictions, remainders and gradient
maxima. The measured positive-step remainder ratio is 3.999751567. Large relative
finite-impact errors remain explicitly visible; no general FV/skill claim is made.

Observed desktop viewport: 1440x900. All three maps loaded and displayed.
The original page was also loaded in a real 390x844 same-origin iframe viewport
in Aside (CSS-responsive check, not native phone/device emulation): document
width 390, panel width 362, map widths 340, no page-wide horizontal overflow.
The numerical table scrolls inside its own container. Screenshots are saved as
`fv_response_desktop.png` and `fv_response_mobile.png` (positive-step panel;
central comparison was pending in these captures).

`page.setViewportSize` is unavailable in this Aside REPL. The explicit iframe
viewport was used instead; it does not change the original page's source.

The publisher confirmed exact preservation of the embedded original data:

- index.html: `5fe7d42540b66e83ca40eb4943a070696402d332f91e4ecea54270ebfb28c31e`
- template.html: `88bb5eb25dfb62cd519d98912d421407678f9957dc7b3a7503444d78a113c8e1`

Raw arrays remain local. Version the small evidence and map assets with the
publisher/template; browser rendering does not rerun assimilation.


After the central run completed, a fresh Aside snapshot confirmed the final
central slope 2.26835255e-3 versus 2.26845231e-3, relative error 4.397e-5,
and endpoint maxima 1.429e-10/2.931e-10. The final table/central text is shown
in `fv_response_central.png`. Table column labels and spacing were improved.
