# System for cyrilic handwritten letter/text recognition
## FINKI, Intelligent Systems
## Team members: Dushica Jankovikj (161509) and Oliver Dimitriov (161535) 

### Project main goal and idea
- Build a system that would recognize cyrillic handwritten letters from images.
- Challenges:
  - drifting handwriting (not in horisontal) and unintelligible handwriting, bad image quality, inappropriate letter background.
  - tokenization of images into sentences, words, letter subparts.

#### Basic approach
- Deep learning approach: multiple neural network architectures tested
- Images containing more than one letter were preprocessed until decomposed to single letter images they contain.
  - Each photo is transformed into a folder of letter images
  - Such single letter images should be easily managed by the letter recognition system

### Dataset
- Russian data set: https://github.com/GregVial/CoMNIST
- Processed and modified so it includes all letters. Example: Manual addition of '\`' to 'к' in order to become 'ќ'.
- Problems: missing letters, upercase dominant, certain letters are more dominant in the dataset - balancing to unifrom distribution was required.

#### Dataset processing and achieving desired pre-train structure
- Organisation into folders (Each letter to its own folder)
- Naming conventions used

### Image processing
#### Dataset preprocessing
- Remove transparency (transform to white)
- Transform each pixel into either black or white depending on a threshold
- Crop surrounding white padding around letter but pay attention to ratio 1:1
- Resize

#### Input image processing (Expected input)
- Row segmentation
- Word segmentation
- Letter segmentation
- Letter segments crop and padding
- Empty spacing recognition

### Deep learning: Neural networks
- Convolutional neural architectures are appropriate for image data. 2D arrays are expected as input.

### Application Flow
- Check if input_image is multiline or not.
- If it is not multi line then it has to be one line or blank.
- If it is blank the application terminates.
- In the remaining cases it process the image the same.
- Separate each line and save each line as new image file.
- Go through each newly line_image and segment characters from the image.
- Make predictions for each line image and save it in a list named final_prediction.
