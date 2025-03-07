import torch
import os 
import argparse
from torchvision import models

# Dictionary to define model architectures with default weights
def get_model_by_name(model_name, num_classes=7):
    if "effnetb0" in model_name:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=num_classes)
        return model

    elif "effnetb2" in model_name:
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        model.classifier[1] = torch.nn.Linear(in_features=1408, out_features=num_classes)
        return model

    elif "vit_b_16" in model_name:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = torch.nn.Linear(in_features=768, out_features=num_classes)
        return model

    else:
        raise ValueError(f"Model architecture for {model_name} is not defined.")

def rename_vit_keys(state_dict, model_name):
    if "vit_b_16" in model_name:
        if 'heads.head.weight' in state_dict:
            print('Key heads.head.weight already found in state_dict')
        elif 'heads.weight' in state_dict:
            print('Renaming heads.weight to heads.head.weight')
            state_dict['heads.head.weight'] = state_dict.pop('heads.weight')
            state_dict['heads.head.bias'] = state_dict.pop('heads.bias')
        elif 'heads.0.weight' in state_dict:
            print('Renaming heads.0.weight to heads.head.weight')
            state_dict['heads.head.weight'] = state_dict.pop('heads.0.weight')
            state_dict['heads.head.bias'] = state_dict.pop('heads.0.bias')
        else:
            print('Key heads.weight or heads.0.weight not found in state_dict')
    return state_dict

def export_model(model, tensor_x, export_method, output_path, opset_version):
    if export_method == 'standard':
        print("Exporting model using torch.onnx.export")
        torch.onnx.export(
            model,
            tensor_x,
            output_path,
            export_params=True,
            opset_version=opset_version, # Compatible opset version
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=None
        )
    elif export_method == 'dynamo':
        print("Exporting using torch.onnx.dynamo_export")
        try:
            onnx_program = torch.onnx.dynamo_export(model, tensor_x)
            onnx_program.save(output_path)
        except Exception as e:
            print(f"Failed to export using dynamo_export: {e}")
            return False
    else:
        raise ValueError("Invalid export method. Choose 'standard' or 'dynamo'.")
    print(f"Model has been successfully exported to {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX.")
    parser.add_argument(
        '--models_folder',
        type=str,
        required=True,
        help='Folder with the PyTorch models'
    )
    parser.add_argument(
        '--export_method',
        type=str,
        choices=['standard', 'dynamo'],
        required=True,
        help='Emthod to export the model: "standard" uses torch.onnx.export, "dynamo" uses torch.onnx.dynamo_export'
    )
    parser.add_argument(
        '--output_folder_standard',
        type=str,
        default='./ONNX_models',
        help='Output folder for standard exported models'
    )
    parser.add_argument(
        '--output_folder_dynamo',
        type=str,
        default='./ONNX_dynamo_export',
        help='Output folder for dynamo exported models'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=7,
        help='Number of output classes for the classification model'
    )

    args = parser.parse_args()

    export_method = args.export_method
    num_classes = args.num_classes

    if export_method == 'standard':
        output_folder = args.output_folder_standard
        opset_version = 14
    else:
        output_folder = args.output_folder_dynamo
        opset_version = 13

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    models_folder = args.models_folder

    # Loop over model files
    for model_filename in os.listdir(models_folder):
        if not model_filename.endswith('.pth'):
            continue # Skip non-PyTorch model files

        model_name = model_filename[:-4] # Remove .pth extension
        print(f'\nLoading model: {model_name}')

        # Initialize model architecture
        try:
            model = get_model_by_name(model_name, num_classes=num_classes)
        except ValueError as ve:
            print(ve)
            continue

        # Load state dict
        state_dict_path = os.path.join(models_folder, model_filename)
        try:
            state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=True) # Load to CPU first
        except Exception as e:
            print(f"Failed to load state dict for {model_name}: {e}")
            continue

        # Rename keys if necessary (for ViT models)
        state_dict = rename_vit_keys(state_dict, model_name)

        # Load state dict into the model
        try:
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Model {model_name} loaded successfully")
        except Exception as e:
            print(f"Failed to load state dict into {model_name}: {e}")
            continue

        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(device)

        # Define output path
        # Save as 'model_name.onnx' instead of 'model_name.pth.onnx'
        onnx_filename = model_name + '.onnx'
        output_path = os.path.join(output_folder, onnx_filename)

        # Export model
        success = export_model(model, dummy_input, export_method, output_path, opset_version)
        if not success:
            print(f"Export failed for {model_name} using {export_method}")
            continue

if __name__ == '__main__':
    main()