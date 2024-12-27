import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
import random
import time

# Initialize model
model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        use_safetensors=True
    )
pipe = pipe.to(device)

if device == "cpu":
    pipe.enable_attention_slicing()

# CSS for enhanced styling
custom_css = """
    .gradio-container {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .gr-box {
        border-radius: 12px !important;
        border: 1px solid #e5e7eb !important;
    }
    .gr-button {
        border-radius: 8px !important;
        background: linear-gradient(to right, #4f46e5, #3b82f6) !important;
        border: none !important;
        color: white !important;
    }
    .gr-button:hover {
        background: linear-gradient(to right, #4338ca, #2563eb) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1) !important;
    }
    .gr-form {
        flex: 1;
        border-radius: 12px !important;
        background: white !important;
        padding: 24px !important;
    }
    .prompt-box {
        border: 2px solid #e5e7eb !important;
        border-radius: 8px !important;
    }
    .prompt-box:focus {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
    }
    .footer {
        margin-top: 24px;
        text-align: center;
        color: #6b7280;
    }
    .loading {
        display: inline-flex;
        gap: 8px;
        align-items: center;
        padding: 12px 24px;
        background: #f3f4f6;
        border-radius: 8px;
        margin: 12px 0;
    }
    .parameters {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 16px 0;
    }
"""

# Example prompts with categories
example_prompts = {
    "Nature": [
        "A misty forest at dawn with sunbeams filtering through the trees",
        "A cascading waterfall in a tropical paradise",
        "Northern lights dancing over a snow-covered mountain range"
    ],
    "Fantasy": [
        "A magical floating castle in the clouds with crystal spires",
        "A mystical dragon perched on a crystal mountain",
        "An enchanted garden with glowing flowers and fairy lights"
    ],
    "Sci-Fi": [
        "A futuristic cyberpunk city at night with neon lights",
        "A space station orbiting a distant planet with multiple moons",
        "Advanced robots working in a high-tech laboratory"
    ]
}

def get_random_prompt():
    category = random.choice(list(example_prompts.keys()))
    return random.choice(example_prompts[category])

def generate_image(prompt, steps, guidance_scale, seed):
    if not prompt:
        return None
    
    try:
        # Show generation status
        gr.Info("🎨 Starting image generation...")
        
        if seed == -1:
            seed = random.randint(0, 2147483647)
        
        generator = torch.Generator(device=device).manual_seed(seed)
        
        image = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        gr.Info("✨ Image generated successfully!")
        return image, seed
    except Exception as e:
        gr.Error(f"Error generating image: {str(e)}")
        return None, seed

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🎨 AI Image Generator")
    gr.Markdown("Transform your imagination into artwork using state-of-the-art AI")
    
    with gr.Row():
        with gr.Column(scale=2):
            # Input area
            prompt = gr.Textbox(
                label="Describe your imagination",
                placeholder="Enter a detailed description of the image you want to create...",
                lines=3,
                elem_classes=["prompt-box"]
            )
            
            with gr.Row():
                submit_btn = gr.Button("🎨 Generate Image", variant="primary")
                random_btn = gr.Button("🎲 Random Prompt")
            
            # Advanced parameters in a collapsible section
            with gr.Accordion("🔧 Advanced Settings", open=False):
                with gr.Row():
                    steps = gr.Slider(minimum=20, maximum=100, value=50, step=1, label="Generation Steps")
                    guidance_scale = gr.Slider(minimum=1, maximum=20, value=7.5, step=0.1, label="Guidance Scale")
                seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
            
            # Example prompts in tabs
            with gr.Tabs():
                for category, prompts in example_prompts.items():
                    with gr.TabItem(f"✨ {category}"):
                        gr.Examples(
                            prompts,
                            inputs=prompt,
                            label=f"{category} Examples"
                        )
        
        with gr.Column(scale=2):
            # Output area
            image_output = gr.Image(label="Generated Image")
            seed_output = gr.Number(label="Seed used", interactive=False)
            
    # Footer
    gr.Markdown(
        """
        <div class="footer">
            <p>🚀 Powered by Stable Diffusion v1.5</p>
            <p>Generate stunning AI art with just a text prompt!</p>
        </div>
        """,
        elem_classes=["footer"]
    )
    
    # Set up event handlers
    submit_btn.click(
        fn=generate_image,
        inputs=[prompt, steps, guidance_scale, seed],
        outputs=[image_output, seed_output]
    )
    
    random_btn.click(
        fn=lambda: gr.update(value=get_random_prompt()),
        outputs=prompt
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()