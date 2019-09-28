# System for cyrilic handwritten letter/text recognition

### Project main goal and idea
- build a system that would recognize cyrillic handwritten letters

#### Initial idea
- build a system that would give textual representation of a given image that contains a cyrilic handwritten text
- Issue: how do i get sentences, words, letters from image

#### Basic approach
- build a system that recognizes handwritten cyrilic letters
  - neural networks used
- preprocess any images containing > 1 letter
  - transform into a folder of letter images
  - letter images should be easily managed by the letter recognition system

### Dataset
- russian data set
- processed and modified dataset so it includes all letters

#### Dataset properties
- size
- organisation into folders
- properties:
  - letters missing
  - uppercase or lowercase dominant
  - uniform distribution or certain letters have more images (disbalance) -> what did we do to improve balance (deleted stufff/ maybe added stuff)
  
#### Dataset processing and achieving desired pre-train structure
- organisation into folders
- naming conventions used

### About neural networks
- general intro
- how do they work
- pros and cons of NN
- for what problems are they applicable 

#### NN usage with specific problems like ours
- other systems that are used for letter recognition
- why use NN for such issues


### Neural network
- layers and why
  - layer type -> why such choice?
  - layer order -> why such order?
- parameters and why
  - test/validation set partition -> why decisions?
  - image resolution -> why such choice?
  
#### History of neural network structure trial and error
- reasons why past versions were worse than current/ optimal
- analysis and conclusions

### Image processing
#### Dataset preprocessing
- remove transparency (transform to white)
- transform each pixel into either black or white depending on a threshold
- crop surrounding white padding around letter but pay attention to ratio 1:1
- resize

#### For input image processing
- row segmentation
- word segmentation
- letter segmentation
- letter segments crop and padding
- empty spacing recognition


### Application Flow
- Check if input_image is multiline or not.
- If it is not multi line then it has to be one line or blank.
- If it is blank the application terminates.
- In the remaining cases it process the image the same.
- Separate each line and save each line as new image file.
- Go through each newly line_image and segment characters from the image.
- Make predictions for each line image and save it in a list named final_prediction.

TODO: 
Comma, Dot ... RECOGNISE THESE? or at least dot?
Documentation related: Make illustrations and schematic representations of NN and problem specific image processing (Dushka)
