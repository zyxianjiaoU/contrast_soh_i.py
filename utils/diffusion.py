import numpy as np
import torch
from utils import nn


def normal_kl(mean1, logvar1, mean2, logvar2):
  """
  KL divergence between normal distributions parameterized by mean and log-variance.
  """
  return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + torch.squared_difference(mean1, mean2) * torch.exp(-logvar2))


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
  betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
  warmup_time = int(num_diffusion_timesteps * warmup_frac)
  betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
  return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
  if beta_schedule == 'quad':
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
  elif beta_schedule == 'linear':
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
  elif beta_schedule == 'warmup10':
    betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
  elif beta_schedule == 'warmup50':
    betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
  elif beta_schedule == 'const':
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
  elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
    betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
  else:
    raise NotImplementedError(beta_schedule)
  assert betas.shape == (num_diffusion_timesteps,)
  return betas


def noise_like(shape, noise_fn=torch.normal, repeat=False, device=None):
    repeat_noise = lambda: torch.repeat(noise_fn(0, 1, size=(1, *shape[1:]), device=device), repeats=shape[0], axis=0)
    noise = lambda: noise_fn(0, 1, size=shape, device=device)
    return repeat_noise() if repeat else noise()


class GaussianDiffusion:
  def __init__(self, *, betas, loss_type='noisepred', torch_dtype=torch.float32):
    self.loss_type = loss_type
    assert isinstance(betas, np.ndarray)
    self.np_betas = betas = betas.astype(np.float64) # computations here in float64 for accuracy
    assert (betas>0).all() and (betas <=1).all()
    timestep, = betas.shape
    self.num_timestep = int(timestep)

    alpha = 1. - betas
    alpha_cumprod = np.cumprod(alpha, axis=0)
    alpha_cumprod_prev = np.append(1., alpha_cumprod[:-1])
    assert alpha_cumprod_prev.shape == (timestep,)

    self.betas = torch.tensor(betas, dtype=torch_dtype)
    self.alpha_cumprod = torch.tensor(alpha_cumprod, dtype=torch_dtype)
    self.alpha_cumprod_prev = torch.tensor(alpha_cumprod_prev, dtype=torch_dtype)

    # calcul for diffusion q(x_t | x_{t-1}) and others
    self.sqrt_alphas_cumprod = torch.tensor(np.sqrt(alpha_cumprod), dtype=torch_dtype)
    self.sqrt_one_minus_alphas_cumprod = torch.tensor(np.sqrt(1 - alpha_cumprod), dtype=torch_dtype)
    self.log_one_minus_alphas_cumprod= torch.tensor(np.log(1. - alpha_cumprod), dtype=torch_dtype)
    self.sqrt_recip_alphas_cumprod = torch.tensor(np.sqrt(1. / alpha_cumprod), dtype=torch_dtype)
    self.sqrt_recipm1_alphas_cumprod = torch.tensor(np.sqrt(1. / alpha_cumprod -1), dtype=torch_dtype)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)
    # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
    self.posterior_variance = torch.tensor(posterior_variance, dtype=torch_dtype)
    # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
    self.posterior_log_variance_clipped = torch.tensor(np.log(np.maximum(posterior_variance, 1e-20)), dtype=torch_dtype)
    self.posterior_mean_coef1 = torch.tensor(betas * np.sqrt(alpha_cumprod_prev) / (1. - alpha_cumprod), dtype=torch_dtype)
    self.posterior_mean_coef2 = torch.tensor((1. - alpha_cumprod_prev) * np.sqrt(alpha) / (1. - alpha_cumprod), dtype=torch_dtype)

  def to(self, device):
    """把扩散系数搬到指定设备（训练/采样必须在同一设备上）。"""
    for name in ['betas', 'alpha_cumprod', 'alpha_cumprod_prev',
                 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                 'log_one_minus_alphas_cumprod', 'sqrt_recip_alphas_cumprod',
                 'sqrt_recipm1_alphas_cumprod', 'posterior_variance',
                 'posterior_log_variance_clipped', 'posterior_mean_coef1',
                 'posterior_mean_coef2']:
      setattr(self, name, getattr(self, name).to(device))
    return self

  @staticmethod
  def _extract(a, t, x_shape):
    """
    Extract some coefficients at specified timesteps,
    then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(input=a, index=t, dim=0)
    assert out.shape[0] == bs
    return torch.reshape(out, [bs]+((len(x_shape) - 1) * [1]))

  def q_mean_variance(self, x_start, t):
    mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
    variance = self._extract(1. - self.alpha_cumprod, t, x_start.shape)
    log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
    return mean, variance, log_variance

  def q_sample(self, x_start, t, noise=None):
    """
    Diffuse the data (t == 0 means diffused for 1 step)
    """
    if noise is None:
      noise = torch.normal(0, 1, size=x_start.shape, device=x_start.device)
    assert noise.shape == x_start.shape
    return (self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
      )

  def predict_start_from_noise(self, x_t, t, noise):
    assert x_t.shape == noise.shape
    return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )

  def q_posterior(self, x_start, x_t, t):
    """
    Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
    """
    assert x_start.shape == x_t.shape
    posterior_mean = (
      self._extract(self.posterior_mean_coef1, t, x_start.shape) * x_start +
      self._extract(self.posterior_mean_coef2, t, x_start.shape) * x_t
    )
    posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
    posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
    assert (posterior_mean.shape[0] == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0] == x_start.shape[0])
    return posterior_mean, posterior_variance, posterior_log_variance_clipped

  def p_losses(self, denoise_fn, x_start, t, noise=None):
    """
    Training loss calculation
    """
    n, c, l = x_start.shape
    assert t.shape[0] == n

    if noise is None:
      noise = torch.normal(0, 1, size=x_start.shape, dtype=x_start.dtype,
                           device=x_start.device)
      assert noise.shape == x_start.shape and noise.dtype == x_start.dtype
      x_noisy = self.q_sample(x_start, t, noise)
      x_recon = denoise_fn(x_noisy, t)
      assert x_noisy.shape == x_start.shape

      if self.loss_type == 'noisepred':
        # predict the noise instead of x_start. seems to be weighted naturally like SNR
        assert x_recon.shape == x_start.shape
        losses = nn.meanflat(torch.square(noise - x_recon))
      else:
        raise NotImplementedError(self.loss_type)
      assert losses.shape[0] == n
      return losses

  def p_mean_variance(self, denoise_fn, *, x, t, clip_denoised: bool):
      if self.loss_type == 'noisepred':
        x_recon = self.predict_start_from_noise(x, t=t, noise=denoise_fn(x, t))
      else:
        raise NotImplementedError(self.loss_type)

      if clip_denoised:
        x_recon = torch.clamp(x_recon, 0, 1)

      model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
      assert model_mean.shape == x_recon.shape == x.shape
      assert list(posterior_variance.shape) == list(posterior_log_variance.shape) == [x.shape[0], 1, 1]
      return model_mean, posterior_variance, posterior_log_variance

  def p_sample(self, denoise_fn, *, x, t, noise_fn, clip_denoised=True, repeat_noise=False):
    """
    Sample from the model
    """
    model_mean, _, model_log_variance = self.p_mean_variance(denoise_fn, x=x, t=t, clip_denoised=clip_denoised)
    noise = noise_like(x.shape, noise_fn, repeat_noise, device=x.device)
    assert noise.shape == x.shape
    # no noise when t = 0
    t0 = (t == torch.tensor(0)).to(torch.int32)
    nonzero_mask = torch.reshape(1 - t0, [x.shape[0]] + [1] * (len(x.shape) - 1))
    return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

  def p_sample_loop(self, denoise_fn, *, shape, noise_fn=torch.normal, device=None):
    """
    Generate samples
    """
    i0 = torch.tensor(self.num_timestep - 1, dtype=torch.int32)
    assert isinstance(shape, (tuple, list))
    img_0 = noise_fn(mean=0, std=1, size=shape, device=device)
    i_, img_ = i0, img_0
    with torch.no_grad():
      while i_ >= 0:
        img_ = self.p_sample(denoise_fn=denoise_fn, x=img_, t=torch.full(size=(shape[0],), fill_value=i_, device=device), noise_fn=noise_fn)
        i_ = i_ - 1
    assert list(img_.shape) == shape
    return img_

  def p_sample_loop_trajectory(self, denoise_fn, *, shape, noise_fn=torch.normal, repeat_noise_steps=-1):
    """
    Generate samples, returning intermediate images
    Useful for visualizing how denoised images evolve over time
    Args:
      repeat_noise_steps (int): Number of denoising timesteps in which the same noise
        is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
    """
    i_0 = torch.tensor(self.num_timestep - 1, dtype=torch.int32)
    assert isinstance(shape, (tuple, list))
    img_0 = noise_like(shape, noise_fn, repeat_noise_steps>=0)
    times = torch.tensor([i_0])
    imgs = torch.tensor([img_0])
    # steps wih repeated noise
    with torch.no_grad():
      while (self.num_timestep - times[-1]) <= repeat_noise_steps:
        times = torch.concatenate([times, [times[-1]-1]], dim=0)
        imgs = torch.concatenate([imgs, [self.p_sample(
          denoise_fn=denoise_fn,
          x=imgs[-1],
          t=torch.full(size=shape[0], fill_value=times[-1]),
          noise_fn=noise_fn,
          repeat_noise=True)]], dim=0)
    # step with different noise for each batch element
    with torch.no_grad():
      while times[-1] >= 0:
        times = torch.concatenate([times, [times[-1] - 1]], dim=0)
        imgs = torch.concatenate([imgs, [self.p_sample(
          denoise_fn=denoise_fn,
          x=imgs[-1],
          t=torch.full([shape[0]], fill_value=times[-1]),
          noise_fn=noise_fn,
          repeat_noise=False)]])
    assert imgs[-1].shape == shape
    return times, imgs

  def interpolate(self, denoise_fn, *, shape, noise_fn=torch.normal):
    """
    Interpolate between images.
    t == 0 means diffuse images for 1 timestep before mixing.
    """
    assert isinstance(shape, (tuple, list))

    # Placeholders for real samples to interpolate
    x1 = torch.Tensor(shape)
    x2 = torch.Tensor(shape)
    # lam == 0.5 averages diffused images
    lam = torch.tensor(0.5)
    t = torch.tensor(100)

    # add noise via forward diffusion
    # todo: use the same noise for both endpoints
    # t_batched = torch.tensor([t] * x1.shape[0], dtype=torch.int32)
    t_batched = torch.stack([t] * x1.shape[0])
    xt1 = self.q_sample(x1, t=t_batched)
    xt2 = self.q_sample(x2, t=t_batched)

    # mix latents
    # linear interpolation
    xt_interp = (1 - lam) * xt1 + lam * xt2
    # Constant variance interpolation
    # xt_interp = torch.sqrt(1 - lam * lam) * xt1 + lam * xt2

    # Reverse diffusion (similar to self.p_sample_loop)
    # t = tf.constant(t, dtype=tf.int32)
    img = xt_interp
    i = t
    with torch.no_grad():
      while i >= 0:
        i = i - 1
        img = self.p_sample(
          denoise_fn=denoise_fn,
          x=img,
          t=torch.full(size=shape[0], fill_value=i),
          noise_fn=noise_fn)
    assert img.shape == shape
    return x1, x2, lam, img, t

  # def training_losses(self):
  #   """
  #   Training loss calculation
  #   """





















