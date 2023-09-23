import pytest
from PIL import Image
from cleanvision import Imagelab

# Create fixtures for the three types of datasets

@pytest.fixture()
def folder_imagelab(generate_local_dataset):
    imagelab = Imagelab(data_path=generate_local_dataset)
    return imagelab

@pytest.fixture()
def hf_imagelab(hf_dataset):
    imagelab = Imagelab(hf_dataset=hf_dataset, image_key="image")
    return imagelab

@pytest.fixture()
def torch_imagelab(torch_dataset):
    imagelab = Imagelab(torchvision_dataset=torch_dataset)
    return imagelab

# Write tests for each dataset type

def test_visualize_local_dataset(folder_imagelab):
    # Visualize random images from the local dataset
    folder_imagelab.visualize()
    
    # Add assertions to check visualization elements (modify as needed)
    assert folder_imagelab.is_grid_displayed() == True
    assert folder_imagelab.are_titles_displayed() == True
    assert folder_imagelab.get_cell_size() == (100, 100)  # Adjust cell size as needed

def test_visualize_hf_dataset(hf_imagelab):
    # Visualize random images from the Hugging Face dataset
    hf_imagelab.visualize()
    
    # Add assertions to check visualization elements (modify as needed)
    assert hf_imagelab.is_grid_displayed() == True
    assert hf_imagelab.are_titles_displayed() == True
    assert hf_imagelab.get_cell_size() == (120, 120)  # Adjust cell size as needed

def test_visualize_torch_dataset(torch_imagelab):
    # Visualize random images from the torchvision dataset
    torch_imagelab.visualize()
    
    # Add assertions to check visualization elements (modify as needed)
    assert torch_imagelab.is_grid_displayed() == True
    assert torch_imagelab.are_titles_displayed() == True
    assert torch_imagelab.get_cell_size() == (80, 80)  # Adjust cell size as needed

if __name__ == '__main__':
    pytest.main()
