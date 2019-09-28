import os
from PIL import Image

def rename_image(input_dir, infile, counter, output_dir = "Dataset"):
    outfile = str(counter)
    extension = os.path.splitext(infile)[1]

    try:
        img = Image.open(input_dir + "/" + infile)
        new_file = output_dir + "/" + outfile + extension
        img.save(new_file)
    except IOError:
        print("Unable to resize image: {}".format(infile))

if __name__ == "__main__":
    output_dir = "Dataset"
    dir = os.getcwd()
    input_dir = "Cyrillic/SH"
    full_input_dir = dir + "/" + input_dir
    full_input_dir_convert = dir + "/" + output_dir

    if not os.path.exists(os.path.join(dir, output_dir)):
        os.mkdir(output_dir)

    try:
        counter = 13063
        for file in os.listdir(full_input_dir):
            rename_image(input_dir, file, counter, output_dir)
            counter += 1

    except OSError:
        print("file not found")