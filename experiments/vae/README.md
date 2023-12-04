### Variational inference on MNIST
This is code we used to train VAE with latents defined over the sphere from section 5.3

### Training
To train embeddings (pole param.: wbgauss, embedded param.: wbgauss_amb):
```
python mnist.py --z_dim Z_DIM --distribution ["wbgauss" | "wbgauss_amb"] --double
```