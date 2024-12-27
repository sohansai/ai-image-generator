# AI Image Generator

This repository provides a simple interface to generate images from text prompts using Stable Diffusion v1.5.

## Features
- Generate high-quality images from text descriptions.
- Easy-to-use Gradio interface for seamless interaction.
- Predefined prompts for Nature, Fantasy, and Sci-Fi categories.
- Customize generation steps, guidance scale, and seed for more control over the output.

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sohansai/ai-image-generator.git
   cd ai-image-generator
   ```
   
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
    ```
   
3. **Run the Application**:
   ```bash
   python app.py
    ```
   
4. **Access the Interface**:

   - Open your browser and navigate to `http://127.0.0.1:7860/`.

   - Enter a text prompt and click "Generate Image" to create your artwork.

## Model
This project uses the Stable Diffusion v1.5 model from Runway ML, optimized for generating high-quality images from text descriptions.

## Requirements
- Python 3.7+
- CUDA-capable GPU (recommended for faster generation)
- Required packages listed in `requirements.txt`
