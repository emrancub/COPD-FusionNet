def generate_map_for_model(model_name, ModelClass, loader):
    model = ModelClass().to(config.DEVICE)
    weight_path = os.path.join(config.CHECKPOINTS_DIR, f"{model_name}_fold0.pth")

    if not os.path.exists(weight_path):
        print(f"Warning: Weights not found for {model_name}. Run train.py first.")
        return

    try:
        state_dict = torch.load(weight_path, map_location=config.DEVICE, weights_only=True)
    except:
        state_dict = torch.load(weight_path, map_location=config.DEVICE, weights_only=False)

    model.load_state_dict(state_dict)

    if "CRNN" in model_name:
        model.train()
        for m in model.modules():
            if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
                m.p = 0
    else:
        model.eval()
    target_input = None
    for i, (inputs, label) in enumerate(loader):
        if i > 50: break

        if label.item() == 1.0: # We want a COPD case
            inputs = inputs.to(config.DEVICE)
            inputs.requires_grad_() # Essential for gradient calculation

            logits = model(inputs)
            prob = torch.sigmoid(logits)

            if prob.item() > 0.80:
                target_input = inputs
                target_label = label
                print(f"  Found sample. Model Confidence: {prob.item():.2%}")
                break

    if target_input is None:
        print("  Could not find a high-confidence positive sample. Skipping.")
        return

    model.zero_grad()

    with torch.backends.cudnn.flags(enabled=False):
        logits = model(target_input)
        logits.backward()

    if target_input.grad is not None:
        saliency = target_input.grad.abs().squeeze().cpu().numpy() # [F, T]
        spectrogram = target_input.detach().squeeze().cpu().numpy() # [F, T]

        # 5. Plotting
        plot_saliency(spectrogram, saliency, model_name)
    else:
        print("  Error: No gradients computed. Model might not support Grad-CAM directly.")


def main():
    loader = load_data_sample()

    # 1. EfficientNet Saliency
    generate_map_for_model("Audio_EffNet", AudioModel_EffNet, loader)

    # 2. CRNN Saliency
    generate_map_for_model("Audio_CRNN", AudioModel_CRNN, loader)

    print(f"\n[Done] Saliency maps generation complete.")

if __name__ == "__main__":
    main()