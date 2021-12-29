import argparse
import os
import uuid
import math

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
import torchvision
from einops.layers.torch import Rearrange, Reduce

from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from gmlp import gMLP, dropout_layers, gMLPBlock, Residual, PreNorm, exists

from vectorquantizer import VectorQuantizerEMA


def train_vq_models(device, name, codebook_size=128):

    batch_size = 96

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.CenterCrop(148),
                                    transforms.Resize(64),
                                    transforms.ToTensor(),
                                    SetRange])

    ds = torchvision.datasets.ImageFolder(root='/home/koepf/data/celeba/', transform=transform) #(root='/data/celeba/', transform=transform)

    def remove_none_collate(batch):
        batch = list(filter(lambda x : x is not None, batch))
        return default_collate(batch)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=remove_none_collate)

    image_size = 64

    #for level in range(0,7):
    for level in range(0,6):
        print('level', level)

        patch_size = image_size // 2**level

        print('patch_size', patch_size)

        num_patches = (image_size // patch_size) ** 2
        model = VectorQuantizerEMA(3*patch_size*patch_size, num_embedding=codebook_size, num_latents=1)

        model.train()
        model.to(device)
        
        init_vector = 0

        to_patch_embed = Rearrange('b c (h p1) (w p2) -> b (h w) c p1 p2', p1 = patch_size, p2 = patch_size)

        step = 1
        next_reuse_step = 500

        for epoch in range(25):

            for t, batch in enumerate(dl, 1):

                batch = batch[0]
                batch_size = batch.size(0)
                batch = batch.to(device)

                batch = to_patch_embed(batch)
                if step == 1:
                    print('batch size after patch split:', batch.size())
                
                #new code
                quantized, encodings, loss, perplexity = model.forward(batch)

                #avg_probs = torch.mean(encodings, dim=0)
                #print('avg_probs', avg_probs[0], avg_probs.size(), avg_probs[0].sum())

                if step % 100 == 0:
                    print('step: {}; loss: {}; perplexity: {}; batch_size: {}; patch_size: {}'.format(step, loss.item(), perplexity.item(), batch_size, patch_size))
                    print(model.activation_count[0].long())

                if step == next_reuse_step:
                    c = model.reuse_inactive()
                    print('resued: ', c)
                    model.reset_stats()
                    next_reuse_step = next_reuse_step + 500
                
                if step % 1000 == 0:
                    model.reset_stats()

                    out_batch = quantized[0]
                    #if (batch_size*num_patches > 1024):
                    #    out_batch = out_batch[0:1024]

                    fn = '{}_quant_{}_{}.png'.format(name, level, step)
                    torchvision.utils.save_image((out_batch + 1.) * 0.5, fn, nrow=int(out_batch.size(0)**0.5), normalize=False)
                    print('saved: ', fn)

                step = step + 1
        
            # save model at end of episode
            fn = '{}_l{}_vq_model_{}.pt'.format(name, level, step)
            data = { 'vq' : model.state_dict() }
            torch.save(data, fn)


def main():
    device_index = 1
    device = torch.device("cuda", device_index)

    vq_model_prefix = 'vq256'
    codebook_size = 256
    #train_vq_models(device, vq_model_prefix, codebook_size=codebook_size)
    num_tokens = codebook_size

    image_size = 64
    
    eval_interval = 1000
    eval_batch_size = 24
    checkpoint_interval = 5000
    eval_loss_interval = 50
    num_eval_iterations = 25
    sample_topk = -1
    eval_noise_schedule = lambda r: r**2

    # parameters
    
    """
    experiment_name = 'masked_denoise_06'
    batch_size = 14
    learning_rate = 0.001
    d_model = 512
    depth = 5
    schedule_name = 'cos3_inv'
    level = 5
    sample_topk = -1
    p_max_uniform = 0.15
    independent_uniform = False
    consistent_masking = False
    weight_decay = 1e-5
    eval_noise_schedule = lambda r: r**4
    use_vq_emb_proj = True
    """

    experiment_name = 'masked_denoise_07'
    batch_size = 14
    learning_rate = 0.0005
    d_model = 512
    depth = 5
    schedule_name = 'cos3_inv'
    level = 5
    sample_topk = -1
    p_max_uniform = 0.1
    independent_uniform = False
    consistent_masking = False
    weight_decay = 1e-7
    eval_noise_schedule = lambda r: r**2
    use_vq_emb_proj = True

    vqs = []
    to_patch_embeds = []

    #checkpoint_name = '.pth'
    checkpoint_name = None
    
    max_levels = 6
    for level in range(max_levels):
        fn = 'vq256/{}_l{}_vq_model_50665.pt'.format(vq_model_prefix, level)
        model_data = torch.load(fn, map_location=device)
        print(fn)
        patch_size = image_size // 2**level
        num_patches = (image_size // patch_size) ** 2

        vq = VectorQuantizerEMA(3*patch_size*patch_size, num_embedding=codebook_size, num_latents=1)
        vq.load_state_dict(model_data['vq'])
        vq.to(device)
        vqs.append(vq)
        
        to_patch_embed = Rearrange('b c (h p1) (w p2) -> b (h w) c p1 p2', p1=patch_size, p2=patch_size)
        to_patch_embeds.append(to_patch_embed)

    # crate dataset / dataloader
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.CenterCrop(148),
                                    transforms.Resize(64),
                                    transforms.ToTensor(),
                                    SetRange])

    ds = torchvision.datasets.ImageFolder(root='/data/celeba/', transform=transform)

    def remove_none_collate(batch):
        batch = list(filter(lambda x : x is not None, batch))
        return default_collate(batch)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=remove_none_collate)
    
    w = 2**level
    if use_vq_emb_proj:
        model = gMLP(num_tokens_in=num_tokens+1, num_tokens_out=num_tokens, vq_embedding_dim=vqs[level].embedding_dim, dim=d_model, depth=depth, seq_len=w*w)
    else:
        model = gMLP(num_tokens_in=num_tokens+1, num_tokens_out=num_tokens, vq_embedding_dim=None, dim=d_model, depth=depth, seq_len=w*w)
    vq_embedding_dim=vqs[level].add_zero_mask_token()

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=False, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25000, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    if checkpoint_name is not None:
        data = torch.load(checkpoint_name, map_location=device)
        model.load_state_dict(data['model_state_dict'])
        optimizer.load_state_dict(data['optimizer_state_dict'])
        lr_scheduler.load_state_dict(data['lr_scheduler_state_dict'])
        print('loaded', checkpoint_name)
    
    def top_k_logits(logits, k):
        v, ix = torch.topk(logits, k, largest=True, sorted=True)
        out = logits.clone()
        out[out < v[:,[-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def evaluate_model(eval_batch_size=24, topk=-1, noise_schedule=None, consistent_masking=False):
        model.eval()
        dec_samples = []

        mask_token_index = num_tokens

        # start with equal weighted probabilities
        logits = torch.zeros(eval_batch_size, w*w, num_tokens, device=device)
        last_mask = torch.zeros(eval_batch_size, w*w) == 0
        
        # start with equal weighted probabilities
        #num_eval_iterations = 100
        for i in range(num_eval_iterations):
            logits_shape = logits.shape

            logits = logits.view(-1, num_tokens)

            # sample
            if topk > 0:
                logits = top_k_logits(logits, topk)
            
            p = F.softmax(logits, dim=-1)
            denoised_samples = torch.multinomial(p, 1, True).view(logits_shape[0], -1)

            frac = (i+1)/num_eval_iterations        

            if noise_schedule is not None:
                alpha = noise_schedule(frac)
            else:
                alpha = frac

            print('alpha', i, alpha)

            if alpha > 1:
                alpha = 1

            if consistent_masking:
                mask = last_mask & (torch.rand(eval_batch_size, w*w) > alpha)
                last_mask = mask
            else:
                mask = (torch.rand(eval_batch_size, w*w) > alpha)
            
            sample = denoised_samples.clone()
            sample[mask] = mask_token_index
            
            # decode 
            #dec = vqs[level].decode(samples)
            dec_denoised = vqs[level].decode(denoised_samples)
            dec = dec_denoised

            # reconstruct image from sample patches
            patch_size = image_size // 2**level
            num_patches = (image_size // patch_size) ** 2
            dec = dec.view(dec.size(0), num_patches, 3, patch_size, patch_size)
            combine_patch = Rearrange('b (h w) c p1 p2 -> b c (h p1) (w p2)', h=w, w=w)
            dec = combine_patch(dec)

            # add to image list
            dec_samples.append(dec)

            # move to more likely sample
            vq_embedding = vqs[level].decode(sample).detach()
            logits = model.forward(sample, vq_embedding) #logits torch.Size([batch_size, 256])

        return torch.cat(dec_samples)

    loss_log = []
    loss_steps = []

    def plot_loss(experiment_name, loss_steps, loss_log):
        batch_size = loss_log[0].size(0)
        #last_step = loss_steps[-1]
        #fn = '{}_plot_{:07}.png'.format(experiment_name, last_step)
        fn = '{}_plot.png'.format(experiment_name)
        
        fig = plt.figure(figsize=(8,8))
    
        ax = fig.add_subplot(1, 1, 1)

        l = torch.cat(loss_log).view(-1, batch_size)
        
        for i in range(0, batch_size, batch_size//10):
            y = l[:,i].view(-1).cpu()
            x = loss_steps
            ax.plot(x, y)

        ax.set_yscale('log')
        ax.set_title('Cross Entropy')
        ax.set_xlabel('iteration')
        fig.savefig(fn, format='png')
        plt.close(fig)

    def named_schedule(name):
        if name == 'linear':
            return lambda r: r
        elif name == 'cos1':
            return lambda r: torch.cos((r + 0.01) / 1.01 * math.pi * 0.5)
        elif name == 'cos2':
            return lambda r: torch.cos((r + 0.01) / 1.01 * math.pi * 0.5) ** 2
        elif name == 'cos05':
            return lambda r: torch.cos((r + 0.01) / 1.01 * math.pi * 0.5) ** 0.5
        elif name == 'cos2_inv':
            return lambda r: 1.0 - torch.cos((r + 0.01) / 1.01 * math.pi * 0.5) ** 2
        elif name == 'cos3_inv':
            return lambda r: 1.0 - torch.cos((r + 0.01) / 1.01 * math.pi * 0.5) ** 3
        elif name == 'cos3':
            return lambda r: torch.cos((r + 0.01) / 1.01 * math.pi * 0.5) ** 3
   
    #dec = evaluate_model(32, topk=-1, noise_schedule=lambda r: r)         
    #torchvision.utils.save_image((dec + 1.) * 0.5, 'eval_test.png', nrow=32, normalize=False)

    mask_token_index = num_tokens

    step = 1
    for epoch in range(25):

        for t, batch in enumerate(dl, 1):
            model.train()

            batch = batch[0]
            batch_size = batch.size(0)
            batch = batch.to(device)

            to_patch_embed = to_patch_embeds[level]
            vq = vqs[level]

            batch_size = batch.size(0)
            b = to_patch_embed(batch)
            encoding = vq.encode(b).view(batch_size, -1)
            
            #inspet batch
            # # decode 
            # dec = vqs[level].decode(encoding)

            # # reconstruct image from sample patches
            # patch_size = image_size // 2**level
            # num_patches = (image_size // patch_size) ** 2
            # dec = dec.view(dec.size(0), num_patches, 3, patch_size, patch_size)
            # combine_patch = Rearrange('b (h w) c p1 p2 -> b c (h p1) (w p2)', h=w, w=w)
            # dec = combine_patch(dec)
            # torchvision.utils.save_image((dec + 1.) * 0.5, 'train_batch.png'.format(experiment_name, step), nrow=eval_batch_size, normalize=False)
            # quit()

            # sizes:
            # encoding torch.Size([24, 256])
            # bs torch.Size([24, 256, 3, 4, 4])
            # batch_size 24

            if step % eval_loss_interval == 0:
                r = torch.linspace(0, 1, batch_size, device=device).view(batch_size, 1)
            else:
                r = torch.rand(batch_size, 1, device=device)
                r = named_schedule(schedule_name)(r)

            threshold = r.to(device)
            mask = torch.rand(batch_size, encoding.size(1), device=device) < threshold

            du = torch.ones(batch_size, encoding.size(1), num_tokens, device=device) / num_tokens
            dt = F.one_hot(encoding, num_classes=num_tokens).float().to(device)
            
            if independent_uniform:
                r = torch.rand(batch_size, 1, device=device) # generate rand independent
            
            d = torch.lerp(dt, du, r.unsqueeze(-1) * p_max_uniform)

            draw = torch.multinomial(d.view(-1, num_tokens), num_samples=1)
            draw = draw.view(batch_size, -1)

            input = draw.clone()
            input[mask] = mask_token_index

            vq_embedding = vq.decode(input).detach()
            logits = model.forward(input, vq_embedding) #logits torch.Size([24, 256, 256])

            loss = loss_fn(logits.view(-1, num_tokens), encoding.view(-1))

            if step % eval_loss_interval == 0:
                per_item_loss = loss.view(batch_size, -1).detach().mean(dim=-1).cpu()
                loss_log.append(per_item_loss)
                loss_steps.append(step)
                plot_loss(experiment_name, loss_steps, loss_log)

            loss = loss.mean()

            if step % 10 == 0:
                print('step: {}; epoche: {}; loss: {}; lr: {};'.format(step, epoch, loss.item(), lr_scheduler.get_last_lr()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if step % eval_interval == 0:
                with torch.no_grad():
                    dec = evaluate_model(eval_batch_size, topk=sample_topk, noise_schedule=eval_noise_schedule, consistent_masking=consistent_masking)
                    # save result as png
                    torchvision.utils.save_image((dec + 1.) * 0.5, '{}_eval_{:07d}.png'.format(experiment_name, step), nrow=eval_batch_size, normalize=False)

            if step % checkpoint_interval == 0:
                fn = '{}_checkpoint_{:07d}.pth'.format(experiment_name, step)
              
                print('writing file: ' + fn)
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'loss_log': loss_log,
                    'loss_steps': loss_steps
                }, fn)

            step += 1


if __name__ == '__main__':
    main()
