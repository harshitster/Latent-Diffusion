import torch

class DDPMConfig:
    def __init__(self):
        self.n_timesteps = None
        self.beta_start = None
        self.beta_end = None 
        self.schedule_type = None

class Kernel:
    def __init__(self):
        self.config = DDPMConfig()

        self.n_timesteps = self.config.n_timesteps
        self.beta_start = self.config.beta_start
        self.beta_end = self.config.beta_end

        if self.config.schedule_type == 'linear':
            self.betas = torch.linspace(self.beta_start, self.beta_end, self.n_timesteps)
        else:
            raise NotImplementedError("Schedule Type Not Implemented")
        
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1. - self.alpha_cum_prod)

    def noise(self, input, noise, t):
        shape = input.shape
        batch_size = shape[0]

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size, *([1] * len(shape[1:])))
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(batch_size, *([1] * len(shape[1:])))

        return sqrt_alpha_cum_prod * input + sqrt_one_minus_alpha_cum_prod * noise
    
    def denoise(self, xt, noise_pred, t):
        x0 = (xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) / self.sqrt_alpha_cum_prod[t]
        x0 = torch.clamp(x0, -1., 1.)

        mean = xt - ((self.betas[t] * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t]))
        mean = mean / torch.sqrt(self.alphas[t])

        if t == 0:
            return mean, x0
        else:
            variance = (1. - self.alpha_cum_prod[t - 1]) / (1. - self.alpha_cum_prod[t])
            variance = variance * self.betas[t]
            sigma = variance ** 0.5

            z = torch.randn(xt.shape)

            return mean + sigma * z, x0