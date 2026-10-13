// Shared helpers of the audiobox_* shaders (MUSIC.md A14c): Audiobox
// Aesthetics' WavLM encoder. Its GELU is torch's exact one, erf and all, so
// it is not kokoro_act's tanh approximation.

// erfc to ~1.2e-7 relative everywhere (Numerical Recipes' erfcc, a Chebyshev
// fit): GLSL has no erf.
float ab_erfc(float x) {
    float z = abs(x);
    float t = 1.0 / (1.0 + 0.5 * z);
    float r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
        + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587
        + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0.0 ? r : 2.0 - r;
}

// gelu(x) = x/2 · (1 + erf(x/√2)) = x/2 · erfc(−x/√2).
float ab_gelu(float x) {
    return 0.5 * x * ab_erfc(-x * 0.70710678118654752);
}
