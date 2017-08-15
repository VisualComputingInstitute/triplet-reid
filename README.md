# Triplet-based Person Re-Identification

Code for reproducing the results of our "In Defense of the Triplet Loss for Person Re-Identification" paper.

Both main authors are currently in an internship.
We will publish the full training code after our internships, which is end of September 2017.
(By "Watching" this project on github, you will receive e-mails about updates to this repo.)
Meanwhile, we provide the pre-trained weights for the TriNet model, as well as some rudimentary example code for using it to compute embeddings, see below.

# Pretrained Models

This is a first, simple release. A better more generic script will follow in a few months, but this should be enough to get started trying out our models!

As a first step, download either of these pre-trained models:
- [TriNet trained on MARS](https://omnomnom.vision.rwth-aachen.de/data/trinet-mars.npz) (md5sum: `72fafa2ee9aa3765f038d06e8dd8ef4b`)
- [TriNet trained on Market1501](https://omnomnom.vision.rwth-aachen.de/data/trinet-market1501.npz) (md5sum: `5353f95d1489536129ec14638aded3c7`)
- (LuNet models will follow.)

Next, create a file (`files.txt`) which contains the full path to the image files you want to embed, one filename per line, like so:

```
/path/to/file1.png
/path/to/file2.jpg
```

Finally, run the `trinet_embed.py` script, passing both the above file and the weights file you want to use, like so:

```
python trinet_embed.py files.txt /path/to/trinet-mars.npz
```

And it will output one comma-separated line for each file, containing the filename followed by the embedding, like so:

```
/path/to/file1.png,-1.234,5.678,...
/path/to/file2.jpg,9.876,-1.234,...
```

You could for example redirect it to a file for further processing:

```
python trinet_embed.py files.txt /path/to/trinet-market1501.npz >embeddings.csv
```

You can now do meaningful work by comparing these embeddings using the Euclidean distance, for example, try some K-means clustering!

A couple notes:
- The script depends on [Theano](http://deeplearning.net/software/theano/install.html), [Lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html) and [OpenCV Python](http://opencv.org/) (`pip install opencv-python`) being correctly installed.
- The input files should be crops of a full person standing upright, and they will be resized to `288x144` before being passed to the network.
