# 2022-09-14 16:10 Seoul
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial17/SimCLR.html

# --- import model ---
from model.Base_encoder import AudioEncoder
import matplotlib.pyplot as plt

# --- import framework ---
import flax 
from flax import jax_utils
from flax.training import train_state, common_utils, checkpoints
import jax
import numpy as np
import jax.numpy as jnp
import optax
from typing import Sequence, Any
import dm_pix


from tqdm import tqdm
import os
import wandb
from utils.config_hook import yaml_config_hook
from functools import partial

class TrainState(train_state.TrainState):
    # Batch statistics from BatchNorm
    batch_stats : Any
    # PRNGKey for augmentations
    rng : Any

    
def augment_image(rng, img):
    rngs = jax.random.split(rng, 7)
    # Random left-right flip
    # Color jitter
    img_jt = img
    img_jt = img_jt * jax.random.uniform(rngs[1], shape=(1,), minval=0.5, maxval=1.5)  # Brightness
    img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
    img_jt = dm_pix.random_contrast(rngs[2], img_jt, lower=0.5, upper=1.5)
    img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
    should_jt = jax.random.bernoulli(rngs[5], p=0.8)
    img = jnp.where(should_jt, img_jt, img)   
    # Gaussian blur
    sigma = jax.random.uniform(rngs[7], shape=(1,), minval=0.5, maxval=1.5)
    img = dm_pix.gaussian_blur(img, sigma=sigma[0], kernel_size=9)
    # Normalization
    img = jax.lax.clamp(0.0, img, 1.0)

    return img

parallel_augment = jax.jit(lambda rng, x: jax.vmap(augment_image)(jax.random.split(rng, x.shape[0]), x))



# --- Define config ---
class TrainerModule:

    def __init__(self,
                 model_name : str,
                 model_class : Any,
                 exmp : Any,
                 lr : float = 5e-4,
                 weight_decay : float = 0.01,
                 seed : int = 33,
                **model_hparams):

        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        # Create empty model. Note: no parameters yet
        self.model_name = model_name
        self.model = model_class(**model_hparams)
        # Prepare logging
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model(exmp)
        wandb.init(
        project='CLR',
        entity='aiffelthon'
        )
    def create_functions(self):
        # To be implemented in sub-classes
        raise NotImplementedError

    def init_model(self, exmp):
        # Initialize model
        rng = jax.random.PRNGKey(self.seed)
        rng, init_rng = jax.random.split(rng)
        variables = self.model.init({'params':init_rng,'dropout':init_rng}, exmp)
        self.state = TrainState(step=0,
                                apply_fn=self.model.apply,
                                params=variables['params'],
                                batch_stats=variables.get('batch_stats'),
                                rng=rng,
                                tx=None, opt_state=None)
        

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        # By default, we decrease the learning rate with cosine annealing
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.lr,
            warmup_steps=0.0,
            decay_steps=int(num_epochs * num_steps_per_epoch),
            end_value=2e-2*self.lr
        )
        optimizer = optax.adamw(lr_schedule, weight_decay=self.weight_decay)
        # optimizer = optax.adam(0.01)
        
        self.create_train_state(optimizer)

    def create_train_state(self, optimizer):
        # Initialize training state
        self.state = TrainState.create(apply_fn=self.state.apply_fn,
                                       params=self.state.params,
                                       batch_stats=self.state.batch_stats,
                                       rng=self.state.rng,
                                       tx=optimizer)

    def train_model(self, train_loader, test_dataloader, num_epochs=200):
        # Train model for defined number of epochs
        # We first need to create optimizer and the scheduler for the given number of epochs
        self.init_optimizer(num_epochs, len(train_loader))
        
        # self.state = jax_utils.replicate(self.state)  # disable pmap
        # Track best eval metric
        for epoch_idx in range(1, num_epochs+1):
            self.train_epoch(train_loader, test_dataloader, epoch=epoch_idx)
        # self.state = jax_utils.unreplicate(self.state) # disable pmap
        
        
    def train_epoch(self, train_dataloader, test_dataloader, epoch):
        # Train model for one epoch, and log avg metrics
        train_iter = iter(train_dataloader)
        test_iter = iter(test_dataloader)
        for n in tqdm(range(len(train_dataloader)), desc=f'Epoch {epoch}', leave=False):
            # train_batch = common_utils.shard(jax.tree_util.tree_map(np.asarray, next(train_iter)))
            # test_batch = common_utils.shard(jax.tree_util.tree_map(np.asarray, next(test_iter)))
            
            self.state, train_metrics, grads, train_cos_sim = self.train_step(self.state, next(train_iter)[0])  # disable pmap
            train_metrics = {k:jax.device_get(v.mean()) for k,v in train_metrics.items()}
            
            self.grads = grads
            
            wandb.log({'train_acc_top1':train_metrics['acc_top1'],
                      'train_acc_top5':train_metrics['acc_top5'],
                      'train_acc_mean_pos':train_metrics['acc_mean_pos'],
                      'train_loss':train_metrics['loss']})

            eval_metrics, eval_cos_sim = self.eval_step(self.state, next(test_iter)[0]) # overwrite batch_metrics
            eval_metrics = {k:jax.device_get(v.mean()) for k,v in eval_metrics.items()}
            wandb.log({'eval_acc_top1':eval_metrics['acc_top1'],
                      'eval_acc_top5':eval_metrics['acc_top5'],
                      'eval_acc_mean_pos':eval_metrics['acc_mean_pos'],
                      'eval_loss':eval_metrics['loss']})
                      
            if n % 100 == 0:
                fig1, ax1 = plt.subplots()
                im1 = ax1.imshow(train_cos_sim, aspect='auto', origin='lower', interpolation='none')
                fig1.colorbar(im1)
                fig1.savefig('train_cos_sim.png')
                plt.close(fig1)

                fig2, ax2 = plt.subplots()
                im2 = ax2.imshow(eval_cos_sim, aspect='auto', origin='lower', interpolation='none')
                fig2.colorbar(im2)
                fig2.savefig('eval_cos_sim.png')
                plt.close(fig2)
                
                wandb.log({'train_cos_sim' : [
                            wandb.Image('train_cos_sim.png')
                            ], 
                           'eval_cos_sim image' : [
                            wandb.Image('eval_cos_sim.png')
                            ]})
                
                
    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir,
                                    target={'params': self.state.params,
                                            'batch_stats': self.state.batch_stats},
                                    step=step,
                                    overwrite=True)

    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        else:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'), target=None)
        num_params = sum([np.prod(p.shape) for p in jax.tree_leaves(state_dict)])
        self.state = TrainState.create(apply_fn=self.state.apply_fn,
                                       params=state_dict['params'],
                                       batch_stats=state_dict['batch_stats'],
                                       rng=self.state.rng,
                                       tx=self.state.tx if self.state.tx else optax.sgd(self.lr)  # Default optimizer
                                      )

    def checkpoint_exists(self):
        # Check whether a pretrained model exist
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'))





class AudioEncoderTrainer(TrainerModule):

    def __init__(self, **kwargs):
        super().__init__(model_name='AudioEncoder',
                         model_class=AudioEncoder,
                         **kwargs)

    def create_functions(self):
        # Function to calculate the InfoNCE loss for a batch of images
        def calculate_loss(params, batch_stats, rng, batch, train):
            # batch = parallel_augment(rng, batch)
            outs = self.model.apply({'params': params, 'batch_stats': batch_stats},
                                    batch, 
                                    rngs={"dropout": rng},
                                    mutable=['batch_stats'] if train else False)
            (loss, metrics, cos_sim), new_model_state = outs if train else (outs, None)
            return loss, (metrics, new_model_state, cos_sim)
        # Training function
        def train_step(state, batch):
            batch = jnp.expand_dims(batch, axis=-1)
            batch = ((batch+100) / 127)
            
            rng, forward_rng = jax.random.split(state.rng)
            batch = parallel_augment(rng, batch)
            
            def loss_fn(params):
                outs = self.model.apply({'params': params, 'batch_stats': state.batch_stats},
                                    batch, 
                                    rngs={"dropout": rng},
                                    mutable=['batch_stats'])
                (loss, metrics, cos_sim), new_model_state = outs
                return loss, (metrics, new_model_state, cos_sim)
            
            
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (_, (metrics, new_model_state, cos_sim)), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads,
                                          batch_stats=new_model_state['batch_stats'],
                                          rng=rng)
            return state, metrics, grads, cos_sim
            
            
            # loss_fn = lambda params: calculate_loss(params,
            #                                         state.batch_stats,
            #                                         forward_rng,
            #                                         batch,
            #                                         train=True)
            # (_, (metrics, new_model_state)), grads = jax.value_and_grad(loss_fn,
            #                                                             has_aux=True)(state.params)
            # # grads = jax.lax.pmean(grads, axis_name='batch')
            # # Update parameters, batch statistics and PRNG key
            # state = state.apply_gradients(grads=grads,
            #                               batch_stats=new_model_state['batch_stats'],
            #                               rng=rng)
            # return state, metrics
            
        # Eval function
        def eval_step(state, batch):
            batch = jnp.expand_dims(batch, axis=-1)            
            batch = ((batch+100) / 127)                        
            rng, forward_rng = jax.random.split(state.rng)         
            batch = parallel_augment(rng, batch)
            
            outs = self.model.apply({'params': state.params, 'batch_stats': state.batch_stats},
                                    batch, 
                                    rngs={"dropout": rng},
                                    mutable=['batch_stats'])
            (loss, metrics, cos_sim), new_model_state = outs
            return metrics, cos_sim
        # jit for efficiency
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)
        
        # self.train_step = jax.pmap(partial(train_step), axis_name='batch')
        # self.eval_step = jax.pmap(partial(eval_step), axis_name='batch')
