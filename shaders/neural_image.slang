// A tiny MLP that maps a pixel position (u, v) to an RGB color, trained to reproduce an image.
//
//   uv (2) -> frequency encoding (ENCODED) -> 32 -> LeakyReLU -> 32 -> LeakyReLU -> 3 -> sigmoid
//
// All weights and biases live in one flat `DiffTensor` so that Slang's autodiff can push
// gradients back into it. Each GPU thread evaluates the network for one pixel; in the backward
// pass the per-pixel gradients are summed into `params.grad_out` (wave reduction + atomic add).

import slangpy;

struct NetworkParameters<int Inputs, int Outputs>
{
    Tensor<float, 1> biases;
    Tensor<float, 2> weights;

    AtomicTensor<float, 1> biases_grad;
    AtomicTensor<float, 2> weights_grad;

    [Differentiable]
    float get_bias(int neuron)
    {
        return biases.load(neuron);
    }

    [Differentiable]
    float get_weight(int neuron, int input)
    {
        return weights.load(neuron, input);
    }

    [BackwardDerivativeOf(get_bias)]
    void get_bias_bwd(int neuron, float grad)
    {
        biases_grad.add(neuron, grad);
    }

    [BackwardDerivativeOf(get_weight)]
    void get_weight_bwd(int neuron, int input, float grad)
    {
        weights_grad.add(neuron, input, grad);
    }

    [Differentiable]
    float[Outputs] forward(float[Inputs] x)
    {
        float[Outputs] y;

        [MaxIters(Outputs)]
        for (int row = 0; row < Outputs; ++row)
        {
            // Sum up bias plus weighted inputs for 1 neuron
            var sum = get_bias(row);
            [ForceUnroll]
            for (int col = 0; col < Inputs; ++col)
                sum += get_weight(row, col) * x[col];
            y[row] = sum;
        }

        return y;
    }
}

struct Network
{
    static const int NumOctaves = 4;
    static const int NumEncodedInputs = 2 * NumOctaves * 2;

    NetworkParameters<NumEncodedInputs, 32> layer0;
    NetworkParameters<32, 32> layer1;
    NetworkParameters<32, 3> layer2;

    // Evaluate the layer and output activations
    [Differentiable]
    float3 eval(no_diff float2 uv)
    {
        // for each input (u and v), we get 2 encoded outputs (sine and cosine) per octave
        float encoded_inputs[2 * NumOctaves * 2];

        [ForceUnroll]
        for (int octave = 0; octave < NumOctaves; ++octave)
        {
            float scale = 2.0f * float.getPi() * float(1 << octave);

            // u encodings
            encoded_inputs[octave * 4 + 0] = sin(uv[0] * scale);
            encoded_inputs[octave * 4 + 1] = cos(uv[0] * scale);

            // v encodings
            encoded_inputs[octave * 4 + 2] = sin(uv[1] * scale);
            encoded_inputs[octave * 4 + 3] = cos(uv[1] * scale);
        }

        float output0[32] = layer0.forward(encoded_inputs);

        // layer 0
        [ForceUnroll]
        for (int i = 0; i < 32; ++i)
            output0[i] = leakyReLU(output0[i]);
        float output1[32] = layer1.forward(output0);

        // layer 1
        [ForceUnroll]
        for (int i = 0; i < 32; ++i)
            output1[i] = leakyReLU(output1[i]);
        float output2[3] = layer2.forward(output1);

        // layer 2
        [ForceUnroll]
        for (int i = 0; i < 3; ++i)
            output2[i] = exp(output2[i]);

        // Return the output (to be used an RGB colour)
        return float3(output2[0], output2[1], output2[2]);
    }
}

[Differentiable]
float ReLU(float x)
{
    return max(x, 0.0f);
}

[Differentiable]
float leakyReLU(float x, float negativeSlope = 0.01f)
{
    if (x > 0.0f)
        return x;
    else
        return x * negativeSlope;
}

// "Render" the neural texture
[Differentiable]
float3 render(int2 pixel, int2 resolution, Network network)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    return network.eval(uv);
}

[Differentiable]
float3 gamma_correct(float3 color)
{
    return pow(color, 1.0f / 2.2f);
}

[Differentiable]
float3 loss(int2 pixel, int2 resolution, no_diff float3 reference, Network network)
{
    float3 color = render(pixel, resolution, network);
    // Colors are in linear color space. Gamma correct before computing loss
    float3 error = gamma_correct(color) - gamma_correct(reference);
    return error * error; // Squared error
}

struct LCG
{
    uint state;

    __init(uint seed)
    {
        state = seed;
    }

    [mutating]
    uint next_uint()
    {
        const uint A = 1664525u;
        const uint C = 1013904223u;
        state = (A * state + C);
        return state;
    }

    [mutating]
    float next_float()
    {
        // Convert to float in range [0, 1)
        return (next_uint() >> 8) * 0x1p-24;
    }
};

void optimizer_step(inout float primal, inout float grad, inout float mean, inout float variance, float learning_rate, int iteration)
{
    // Default Adam values.
    const float ADAM_BETA_1 = 0.9;
    const float ADAM_BETA_2 = 0.999;
    const float ADAM_EPSILON = 1e-8;

    // Running average of gradient and square gradient (== variance)
    mean = lerp(grad, mean, ADAM_BETA_1);
    variance = lerp(grad * grad, variance, ADAM_BETA_2);

    // The running average is initialized with zeros, and underestimates
    // the true values at the beginning. Adam divides by a compensation factor
    float mHat = mean / (1.0f - pow(ADAM_BETA_1, iteration));
    float vHat = variance / (1.0f - pow(ADAM_BETA_2, iteration));
    // Compute the adam step by scaling the mean gradient by the learning rate and the variance estimate
    float step = learning_rate * mHat / (sqrt(vHat) + ADAM_EPSILON);

    // Take an adam step
    primal -= step;

    // Because we now accumulate gradients, we need to clear the gradient to zero before the next iteration
    grad = 0.0f;
}

void calculate_grads(uint seed, int2 batch_index, int2 batch_size, Tensor<float3, 2> reference, Network network)
{
    // Pick a random pixel for computing the loss
    // First, sample a random UV within our cell of the training batch grid
    LCG lcg = LCG(seed);
    float2 uv = (batch_index + float2(lcg.next_float(), lcg.next_float())) / float2(batch_size);
    // Then turn that into a pixel on the texture
    int2 resolution = { reference.shape[1], reference.shape[0] };
    int2 pixel = int2(uv * float2(resolution));
    // Evaluate the reference texture color there
    float3 reference_color = reference.load(pixel);
    // Backpropagate through the loss
    bwd_diff(loss)(pixel, resolution, reference_color, network, 1.0f);
}
