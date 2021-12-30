CHAR_VECTOR = "456790BFHJKLMNWVTXPR"

letters = [letter for letter in CHAR_VECTOR]

num_classes = len(letters) + 1
img_w, img_h = 128, 128

# Network parameters
batch_size = 16
val_batch_size = 8

downsample_factor = 4
max_text_len = 10

