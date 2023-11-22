def generate_image(job):
    '''
    Generate an image from text using Kandinsky3
    '''
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    generator = _setup_generator(validated_input['seed'])

    # Run inference on the model and get the image embeddings
    image_embeds, negative_image_embeds = pipe_prior_3(
        validated_input['prompt'],
        validated_input['negative_prompt'],
        generator=generator).to_tuple()

    init_image = None
    if job_input.get('init_image', None) is not None:
        init_image = load_image(job_input['init_image'])

    image_urls = []
    if init_image is None:
        output = pipe_prior_3(
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            height=validated_input['h'],
            width=validated_input['w'],
            num_inference_steps=validated_input['num_steps'],
            guidance_scale=validated_input['guidance_scale'],
            num_images_per_prompt=validated_input['num_images'],
            generator=generator).images
    else:
        output = pipe_prior_3(
            validated_input["prompt"],
            image=init_image,
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            height=validated_input['h'],
            width=validated_input['w'],
            num_inference_steps=validated_input['num_steps'],
            strength=validated_input['strength']).images
        
    image_urls = _save_and_upload_images(output, job['id'])

    return {"image_url": image_urls[0]} if len(image_urls) == 1 else {"images": image_urls}
