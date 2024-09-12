from .smplx_face import TrainWrapper as s2g_face
from .smplx_body_vq import TrainWrapper as s2g_body_vq
from .smplx_body_pixel import TrainWrapper as s2g_body_pixel
from .smplx_gestformer_vqvae import TrainWrapper as gestformer_encoder

from .body_ae import TrainWrapper as s2g_body_ae
from .LS3DCG import TrainWrapper as LS3DCG
from .base import TrainWrapperBaseClass

from .utils import normalize, denormalize