from backbones.config import Config as ModelConfig
from dataset.config import Config as DataConfig


class TrainConfig:
    """Configuration for training loop."""

    def __init__(self):
        # choose "fixed"  or  "cosine_restart or "warmup_cosine"  "two_phase_cosine"
        self.lr_policy     = "fixed"
        self.learning_rate = 1e-4       # base LR  (ignored when cosine sets its own)
        self.warmup_steps  = 2_000          # optional warm‑up inside cosine
        self.restart_epochs = 30            # cosine restart period
        # warmup_cosine
        self.warmup_initial_lr = 1e-6
        self.warmup_epochs = 3
        self.cosine_floor_factor = 1e-2
        # Adam hyper‑params
        self.ema_decay = 0.995
        self.beta1 = 0.9
        self.beta2 = 0.98 # 0.99 for UNet
        self.eps   = 1e-8 # 1e-9
        self.weight_decay = 0.0

        # cold diffusion params
        self.diffusions_steps = 16  #10 default
        self.diffusion_mode = 'linear' # linear, sqrt_pair, sqrt_aggresive
        self.alpha_mode = 'cos2' # poly, cos2, exp, sigmoid
        self.residual_mode  = 'none'  #next_delta_norm, clean_residual or none (direct spec generation)

        
        # global params
        self.epochs = 200
        self.patience = 7
        self.gen_val_batch = True # generate random val batch after epoch


class Config:
    """Integrated configuration."""

    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.train = TrainConfig()

    def dump(self):
        """Dump configurations into serializable dictionary."""
        return {k: vars(v) for k, v in vars(self).items()}

    @staticmethod
    def load(dump_):
        """Load dumped configurations into new configuration."""
        conf = Config()
        for k, v in dump_.items():
            if hasattr(conf, k):
                obj = getattr(conf, k)
                load_state(obj, v)
        return conf


def load_state(obj, dump_):
    """Load dictionary items to attributes."""
    for k, v in dump_.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    return obj
