"""Train a tiny MLP to reproduce a image.

The network, its loss, backpropagation (via Slang autodiff) and the Adam optimizer all run on
the GPU in `shaders/neural_image.slang`; Python only owns the buffers and the training loop.

    uv run python experiments/neural_image.py
    uv run python experiments/neural_image.py --image path/to/photo.png --steps 3000
    uv run python experiments/neural_image.py --frequencies 0   # no encoding: see it go blurry
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import slangpy as spy
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from tensorboardX import SummaryWriter

from app import App

# Create the app and load the slang module.
app = App(width=512 * 3 + 10 * 2, height=512, title="Neural Image")
module = spy.Module.load_from_file(app.device, "neural_image.slang")

# Load some materials.
data_path = Path(__file__).resolve().parents[1] / "images"
image = spy.Tensor.load_from_image(app.device, data_path.joinpath("slangstars.png"), linearize=True)


class NetworkParameters(spy.InstanceList):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(module[f"NetworkParameters<{inputs},{outputs}>"])
        self.inputs = inputs
        self.outputs = outputs

        # Biases and weights for the layer.
        self.biases = spy.Tensor.from_numpy(app.device, np.zeros(outputs).astype("float32"))
        self.weights = spy.Tensor.from_numpy(
            app.device, np.random.uniform(-0.5, 0.5, (outputs, inputs)).astype("float32")
        )

        # Gradients for the biases and weights.
        self.biases_grad = spy.Tensor.zeros_like(self.biases)
        self.weights_grad = spy.Tensor.zeros_like(self.weights)

        # Temp data for Adam optimizer.
        self.m_biases = spy.Tensor.zeros_like(self.biases)
        self.m_weights = spy.Tensor.zeros_like(self.weights)
        self.v_biases = spy.Tensor.zeros_like(self.biases)
        self.v_weights = spy.Tensor.zeros_like(self.weights)

    # Calls the Slang 'optimize' function for biases and weights
    def optimize(self, learning_rate: float, optimize_counter: int):
        module.optimizer_step(
            self.biases,
            self.biases_grad,
            self.m_biases,
            self.v_biases,
            learning_rate,
            optimize_counter,
        )
        module.optimizer_step(
            self.weights,
            self.weights_grad,
            self.m_weights,
            self.v_weights,
            learning_rate,
            optimize_counter,
        )


class LatentTexture(spy.InstanceList):
    def __init__(self, width: int, height: int, num_latents: int):
        super().__init__(module[f"LatentTexture<{num_latents}>"])

        # Grid of learned feature vectors, indexed (y, x, latent).
        self.texture = spy.Tensor.from_numpy(
            app.device,
            np.random.uniform(-0.5, 0.5, (height, width, num_latents)).astype("float32"),
        )
        self.texture_grads = spy.Tensor.zeros_like(self.texture)

        # Temp data for Adam optimizer.
        self.m_texture = spy.Tensor.zeros_like(self.texture)
        self.v_texture = spy.Tensor.zeros_like(self.texture)

    # Calls the Slang 'optimize' function for the latents
    def optimize(self, learning_rate: float, optimize_counter: int):
        module.optimizer_step(
            self.texture,
            self.texture_grads,
            self.m_texture,
            self.v_texture,
            learning_rate,
            optimize_counter,
        )


# Must match the constants in the Slang `Network` struct.
NUM_OCTAVES = 4
NUM_FREQ_INPUTS = 2 * NUM_OCTAVES * 2
NUM_LATENTS = 4
NUM_HIDDEN = 64
STEPS_PER_FRAME = 100


class Network(spy.InstanceList):
    def __init__(self):
        super().__init__(module["Network"])
        self.latent_texture = LatentTexture(64, 64, NUM_LATENTS)
        self.layer0 = NetworkParameters(NUM_FREQ_INPUTS + NUM_LATENTS, NUM_HIDDEN)
        self.layer1 = NetworkParameters(NUM_HIDDEN, NUM_HIDDEN)
        self.layer2 = NetworkParameters(NUM_HIDDEN, 3)

    # Calls the Slang 'optimize' function for the latents and each layer.
    def optimize(self, learning_rate: float, optimize_counter: int):
        self.latent_texture.optimize(learning_rate, optimize_counter)
        self.layer0.optimize(learning_rate, optimize_counter)
        self.layer1.optimize(learning_rate, optimize_counter)
        self.layer2.optimize(learning_rate, optimize_counter)


network = Network()

optimize_counter = 0

# Slang will compile the shaders the first time we call into them (i.e. in the first iteration)
print("Compiling shaders... this may take a while")
progress = Progress(
    SpinnerColumn(),
    TextColumn("[bold]Training"),
    TextColumn("{task.completed:>10,.0f} steps"),
    TextColumn("loss [cyan]{task.fields[loss]:.5f}"),
    TextColumn("best [green]{task.fields[best]:.5f}"),
    TimeElapsedColumn(),
)
task = progress.add_task("train", total=None, loss=float("nan"), best=float("nan"))
best_loss = float("inf")
progress.start()

# Log metrics for TensorBoard; view with `uv run tensorboard --logdir outputs/runs`.
run_dir = Path(__file__).resolve().parents[1] / "outputs" / "runs"
writer = SummaryWriter(str(run_dir / datetime.now().strftime("%Y%m%d-%H%M%S")))

while app.process_events():
    offset = 0
    # Blit reference texture
    app.blit(image, size=spy.int2(512), offset=spy.int2(offset, 0), tonemap=False, bilinear=True)
    offset += 512 + 10
    res = spy.int2(image.shape[1], image.shape[0])
    # Train a batch of samples at a time. Smaller batches train faster, but are more "jittery"
    # A better strategy is to use small batches at the start, and slowly increase them over time
    batch_size = (64, 64)

    # Render current neural texture
    lr_output = spy.Tensor.empty_like(image)
    module.render(pixel=spy.call_id(), resolution=res, network=network, _result=lr_output)
    app.blit(
        lr_output, size=spy.int2(512, 512), offset=spy.int2(offset, 0), tonemap=False, bilinear=True
    )
    offset += 512 + 10

    # Show loss between neural texture and reference texture.
    loss_output = spy.Tensor.empty_like(image)
    module.loss(
        pixel=spy.call_id(), resolution=res, network=network, reference=image, _result=loss_output
    )
    app.blit(loss_output, size=spy.int2(512, 512), offset=spy.int2(offset, 0), tonemap=False)

    learning_rate = 0.001

    # NOTE: For faster feedback we use a low number of iterations (20) per frame. This will be more efficient if you turn it up.
    for i in range(STEPS_PER_FRAME):
        module.calculate_grads(
            seed=spy.wang_hash(seed=optimize_counter, warmup=2),
            batch_index=spy.grid(batch_size),
            batch_size=spy.int2(batch_size),
            reference=image,
            network=network,
        )
        optimize_counter += 1

        network.optimize(learning_rate, optimize_counter)

    loss = float(np.mean(loss_output.to_numpy()))
    best_loss = min(best_loss, loss)
    progress.update(task, advance=STEPS_PER_FRAME, loss=loss, best=best_loss)

    writer.add_scalar("loss", loss, optimize_counter)
    writer.add_scalar("best_loss", best_loss, optimize_counter)
    if optimize_counter % 5000 == 0:
        snapshot = lr_output.to_numpy()[..., :3]
        snapshot = np.clip(snapshot, 0.0, 1.0) ** (1.0 / 2.2)  # linear -> roughly sRGB for display
        writer.add_image("reconstruction", snapshot, optimize_counter, dataformats="HWC")

    # Present the window.
    app.present()

progress.stop()
writer.close()
