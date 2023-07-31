from math import pi

import pytorch_lightning as pl
from hydra_zen import builds, make_config, make_custom_builds_fn, zen
import torch as tr
from torch.optim import Adam
from torch.utils.data import DataLoader


pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)


ExperimentConfig = make_config(
    seed=1,
    lit_module=pl.LightningModule,
    trainer=builds(pl.Trainer, max_epochs=100),
    model=builds(single_layer_nn, num_neurons=10),
    optim=pbuilds(Adam),
    dataloader=pbuilds(DataLoader, batch_size=25, shuffle=True, drop_last=True),
    target_fn=tr.cos,
    training_domain=builds(tr.linspace, start=-2 * pi, end=2 * pi, steps=1000),
)

# Wrapping `train_and_eval` with `zen` makes it compatible with Hydra as a task function
#
# We must specify `pre_call` to ensure that pytorch lightning seeds everything
# *before* any of our configs are instantiated (which will initialize the pytorch
# model whose weights depend on the seed)
pre_seed = zen(lambda seed: pl.seed_everything(seed))
task_function = zen(train_and_eval, pre_call=pre_seed)

if __name__ == "__main__":
    # enables us to call
    from hydra_zen import ZenStore

    store = ZenStore(deferred_hydra_store=False)
    store(ExperimentConfig, name="lit_app")

    task_function.hydra_main(
        config_name="lit_app",
        version_base="1.1",
        config_path=".",
    )