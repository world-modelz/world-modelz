# som-diffusion

## Idea
Use a (square) 2D [Self-organizing map](https://en.wikipedia.org/wiki/Self-organizing_map) (SOM) instead of the unordered vector quantization (VQ) codebook to quantize the latent representation of an auto-encoder (similar to VQ-VAE). Convert index of best matching unit (BMU) into a normalized 2D vector by computing its 2D x/y offset to the map-center coordinate and divide by map-width/2 to constraint the x,y components into [-1, 1]. Use this vector representation as input to a denoising diffusion process. To decode images sample a latent representation via diffusion backward-process (denoising), convert resulting vector into SOM map positions and pass the vector of the BMU (potentially interpolated) through the auto-encoder decoder network to retrieve the output image.

## Plan / progress
- [OK] implement batched SOM layer as PyTorch nn.Module 
- [OK] build encoder/decoder pair for auto-encoder and train on imagenet_small 64x64
- [OK] train SOM with image-net encoder outputs
- [OK] verify that full roundtrip over 2D SOM positions works
- [0%] measure som distances to neighbors
- [25%] build denoising model and train on SOM output vectors 

## potential experiments
- growing som vs. final SOM size with shrinking gaussian kernel
- first learn auto-encoder alone (without quantization) and start learning the SOM only after good auto-encoder representations have been formed, during SOM training freeze weights of AE, potentially in final phase learn everything together 

## References
- Paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239); Repo: [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)
- Paper: [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672); Repo: [openai/improved-diffusion](https://github.com/openai/improved-diffusion)
