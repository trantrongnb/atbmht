import torch
from diffusers import AutoPipelineForText2Image

def test_gradient():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model...")
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to(device)
    
    # Freeze
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    
    # Prompt
    prompt = "a colorful abstract painting"
    prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False
    )
    add_time_ids = pipe._get_add_time_ids(
        (512, 512), (0, 0), (512, 512), dtype=prompt_embeds.dtype, text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim
    ).to(device)
    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
    
    pipe.scheduler.set_timesteps(1, device=device)
    timesteps = pipe.scheduler.timesteps
    t = timesteps[0]
    
    latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16, requires_grad=True)
    
    latent_model_input = latents * pipe.scheduler.init_noise_sigma
    
    noise_pred = pipe.unet(
        latent_model_input,
        t,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]
    
    # Scheduler step
    denoised_latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    
    # Decode
    image = pipe.vae.decode(denoised_latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    
    loss = image.sum()
    loss.backward()
    
    print("Gradient norms:", latents.grad.norm())
    print("Test passed!")

if __name__ == "__main__":
    test_gradient()
