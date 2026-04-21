import torch
from llava.model.builder import load_pretrained_model as load_llava_model
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token
from transformers import CLIPImageProcessor


def _decode_generation(tokenizer, generated_ids, input_ids):
    """Decode only newly generated tokens when generate() returns prompt+answer."""
    input_len = input_ids.shape[1]
    decode_ids = generated_ids

    if generated_ids.shape[1] > input_len:
        generated_prefix = generated_ids[:, :input_len].to(input_ids.device)
        comparable = input_ids != IMAGE_TOKEN_INDEX
        if comparable.any():
            match_rate = (
                generated_prefix[comparable] == input_ids[comparable]
            ).float().mean().item()
            if match_rate > 0.95:
                decode_ids = generated_ids[:, input_len:]

    return tokenizer.batch_decode(decode_ids, skip_special_tokens=True)[0].strip()


def _join_prefix(prefix, continuation):
    if not prefix:
        return continuation
    if not continuation:
        return prefix
    if continuation[0] in " \n\t,.;:!?)]}":
        return f"{prefix}{continuation}".strip()
    return f"{prefix} {continuation}".strip()


def model_inference(engine, model, tokenizer, image, prompt, processor, max_new_tokens, assistant_prefix=None):
    
    # Build a processor if one wasn't provided by the loader
    if processor is None:
        vision_tower = getattr(model.config, 'mm_vision_tower', 'openai/clip-vit-large-patch14-336')
        processor = CLIPImageProcessor.from_pretrained(vision_tower)

    # Support both legacy `.preprocess` and callable processors
    if hasattr(processor, 'preprocess'):
        processed = processor.preprocess([image], return_tensors='pt')
    else:
        processed = processor(images=[image], return_tensors='pt')

    image_tensor = processed['pixel_values'].to(torch.float16).cuda()
    
    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    
    conv_mode = 'llava_v1'
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    assistant_prefix = assistant_prefix.strip() if assistant_prefix else None
    if assistant_prefix:
        prompt = f"{prompt} {assistant_prefix}"

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0),
            do_sample=False,
            temperature=1,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            )
    predicted_answers = _decode_generation(tokenizer, generated_ids, input_ids)
    return _join_prefix(assistant_prefix, predicted_answers)

def load_model(model_path, model_base=None, model_name='llava', args=None):
    tokenizer, model, image_processor, context_len = load_llava_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        attn_implementation='flash_attention_2',
        torch_dtype='float16',
        device_map='cuda',
    )
    # Ensure the vision tower is loaded and a processor exists
    try:
        vt = model.get_vision_tower()
        if hasattr(vt, 'is_loaded') and not vt.is_loaded:
            vt.load_model(device_map='cuda')
        if image_processor is None and hasattr(vt, 'image_processor'):
            image_processor = vt.image_processor
    except Exception:
        # Fallback to constructing from config if accessing the tower fails
        if image_processor is None:
            vision_tower = getattr(model.config, 'mm_vision_tower', 'openai/clip-vit-large-patch14-336')
            image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
    processor = image_processor
    return model, tokenizer, processor
