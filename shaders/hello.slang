// Minimal example functions callable from Python via slangpy.

float add(float a, float b)
{
    return a + b;
}

// Differentiable polynomial: slangpy can call bwd_diff(polynomial) for gradients.
[Differentiable]
float polynomial(float a, float b, float c, float x)
{
    return a * x * x + b * x + c;
}

// Procedural image: one call per pixel when invoked with a grid of coordinates.
float3 checkerboard(int2 pixel, int2 size, int tiles)
{
    float2 uv = float2(pixel) / float2(size);
    int2 cell = int2(uv * tiles);
    float v = ((cell.x + cell.y) & 1) == 0 ? 1.0 : 0.15;
    return float3(uv.x, uv.y, 1.0 - uv.x) * v;
}
