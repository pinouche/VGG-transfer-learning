Transfer learning using the VVG 16 network. Images of different sizes can be fed into the VGG.
The features obtained come from the last average pooling layer of the VGG and only the weights of the 2 outer fully connected layers are updated/trained.

This is for a specific problems where we had to manually entered the labels manually. The key function for transfer learning where we feed the new images and fetch the last average pool layer of the VGG is utils.Get_ResizedImagesTrain(new_width, new_height, path, vgg_layers).
